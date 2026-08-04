/* crs.c — the certify() decode kernel in C.
 *
 * Exact port of reedsolo's rs_correct_msg (prim 0x11d, generator 2, fcr 0)
 * plus the erasure ladder from softdec.certify(): try the hard decode, then
 * erase the n_er least-confident bytes for n_er in {4, 10, .., <=0.7*ecc}.
 * The caller passes the confidence ORDER (numpy argsort output), so
 * tie-breaking is bit-identical to the Python path by construction.
 *
 * CRC32 (zlib polynomial) over the decoded block closes the certification,
 * so one call does what certify()'s inner loop did in ~30 Python-level RS
 * decodes. Build: cc -O2 -shared -o crs.dylib crs.c
 */
#include <stdint.h>
#include <string.h>

static uint8_t GF_EXP[512];
static uint8_t GF_LOG[256];
static uint32_t CRC_T[256];
static int INIT_DONE = 0;

static void init_tables(void) {
    if (INIT_DONE) return;
    int x = 1;
    for (int i = 0; i < 255; i++) {
        GF_EXP[i] = (uint8_t)x;
        GF_LOG[x] = (uint8_t)i;
        x <<= 1;
        if (x & 0x100) x ^= 0x11d;
    }
    for (int i = 255; i < 512; i++) GF_EXP[i] = GF_EXP[i - 255];
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        CRC_T[n] = c;
    }
    INIT_DONE = 1;
}

static inline uint8_t gmul(uint8_t a, uint8_t b) {
    if (!a || !b) return 0;
    return GF_EXP[GF_LOG[a] + GF_LOG[b]];
}
static inline uint8_t gdiv(uint8_t a, uint8_t b) {
    if (!a) return 0;
    return GF_EXP[(GF_LOG[a] + 255 - GF_LOG[b]) % 255];
}
static inline uint8_t gpow(int p) {
    int e = p % 255; if (e < 0) e += 255;
    return GF_EXP[e];
}

uint32_t crc32_c(const uint8_t *buf, int len) {
    init_tables();
    uint32_t c = 0xFFFFFFFFu;
    for (int i = 0; i < len; i++)
        c = CRC_T[(c ^ buf[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

/* syndromes: synd[0]=0 pad, synd[1..nsym] = eval(msg, alpha^i), i=0.. */
static int calc_synd(const uint8_t *msg, int n, int nsym, uint8_t *synd) {
    int nz = 0;
    synd[0] = 0;
    for (int i = 0; i < nsym; i++) {
        uint8_t a = gpow(i), y = 0;
        for (int j = 0; j < n; j++) y = gmul(y, a) ^ msg[j];
        synd[i + 1] = y;
        nz |= y;
    }
    return nz;   /* 0 means clean */
}

/* One RS(255) codeword: hard decode, then the erasure ladder.
 * chunk: 255 bytes (modified in place on success).
 * order: 255 indices, least-confident first (numpy argsort of byteconf).
 * Returns 1 and writes the corrected 255 bytes back on success, else 0. */
static int correct(const uint8_t *in, int nsym, const int32_t *erase,
                   int n_erase, uint8_t *out) {
    uint8_t msg[255], synd[49], fsynd[49];
    uint8_t err_loc[50], old_loc[50], tmp[50];
    int n = 255;
    if (n_erase > nsym) return 0;
    memcpy(msg, in, n);
    for (int i = 0; i < n_erase; i++) msg[erase[i]] = 0;
    if (!calc_synd(msg, n, nsym, synd)) { memcpy(out, msg, n); return 1; }

    /* Forney syndromes: fold the erasures out */
    int fn = nsym - n_erase;
    {
        uint8_t fs[49];
        for (int i = 0; i <= nsym; i++) fs[i] = synd[i];
        for (int i = 0; i < n_erase; i++) {
            uint8_t x = gpow(n - 1 - erase[i]);
            for (int j = 0; j < nsym - (i + 1) + 1 - 1; j++)
                fs[1 + j] = gmul(fs[1 + j], x) ^ fs[1 + j + 1];
        }
        for (int i = 0; i <= fn; i++) fsynd[i] = fs[i];
    }

    /* Berlekamp-Massey on the Forney syndromes */
    int el_len = 1, ol_len = 1;
    err_loc[0] = 1; old_loc[0] = 1;
    for (int i = 0; i < fn; i++) {
        uint8_t delta = fsynd[i + 1];
        for (int j = 1; j < el_len; j++)
            delta ^= gmul(err_loc[el_len - 1 - j], fsynd[i + 1 - j]);
        old_loc[ol_len++] = 0;
        if (delta) {
            if (ol_len > el_len) {
                for (int j = 0; j < ol_len; j++)
                    tmp[j] = gmul(old_loc[j], delta);
                for (int j = 0; j < el_len; j++)
                    old_loc[ol_len - el_len + j] = gdiv(err_loc[j], delta);
                memcpy(err_loc, tmp, ol_len); el_len = ol_len;
                memcpy(old_loc + (ol_len - el_len), err_loc, 0); /* no-op */
            } else {
                for (int j = 0; j < ol_len; j++)
                    err_loc[el_len - ol_len + j] ^= gmul(old_loc[j], delta);
            }
        }
    }
    /* strip leading zeros */
    int lead = 0;
    while (lead < el_len && err_loc[lead] == 0) lead++;
    int L = el_len - lead;
    uint8_t *eloc = err_loc + lead;
    int errs = L - 1;
    if (errs * 2 + n_erase > nsym) return 0;

    /* Chien search over message positions */
    int err_pos[49], n_err = 0;
    for (int i = 0; i < n; i++) {
        uint8_t x = gpow(i), y = 0;
        for (int j = 0; j < L; j++) y = gmul(y, x) ^ eloc[j];
        if (y == 0) {
            if (n_err >= 49) return 0;
            err_pos[n_err++] = n - 1 - i;
        }
    }
    if (n_err != errs) return 0;

    /* errata = erasures + errors; Forney correction on the ORIGINAL synd */
    int pos[98], np_ = 0;
    for (int i = 0; i < n_erase; i++) pos[np_++] = erase[i];
    for (int i = 0; i < n_err; i++) pos[np_++] = err_pos[i];

    /* errata locator from positions */
    uint8_t loc[99]; int loc_len = 1; loc[0] = 1;
    for (int i = 0; i < np_; i++) {
        uint8_t x = gpow(n - 1 - pos[i]);
        /* loc *= (1 + x*z)  -> poly mult by [x, 1] */
        uint8_t nl[100];
        nl[0] = gmul(loc[0], x);
        for (int j = 1; j < loc_len; j++)
            nl[j] = gmul(loc[j], x) ^ loc[j - 1];
        nl[loc_len] = loc[loc_len - 1];
        loc_len++;
        memcpy(loc, nl, loc_len);
    }
    /* evaluator = (synd_rev * loc) mod z^(np_+1) ; reedsolo uses
       rs_find_error_evaluator(synd[::-1 minus pad], ...) — implement via
       direct convolution of reversed syndromes */
    uint8_t srev[49];
    for (int i = 0; i < nsym; i++) srev[i] = synd[nsym - i];  /* drop pad */
    uint8_t ev[99]; int ev_len = np_ + 1;
    for (int i = 0; i < ev_len; i++) {
        uint8_t acc = 0;
        for (int j = 0; j <= i; j++)
            if (j < nsym && (i - j) < loc_len)
                acc ^= gmul(srev[nsym - 1 - j], loc[loc_len - 1 - (i - j)]);
        ev[ev_len - 1 - i] = acc;
    }

    memcpy(out, msg, n);
    for (int i = 0; i < np_; i++) {
        int p = pos[i];
        uint8_t xi = gpow(n - 1 - p);
        uint8_t xi_inv = gdiv(1, xi);
        /* formal derivative of loc at xi_inv: odd-power terms */
        uint8_t den = 0;
        for (int j = loc_len - 2; j >= 0; j -= 2) {
            /* coefficient loc[j] multiplies z^(loc_len-1-j); odd powers */
            int power = loc_len - 1 - j;
            if (power & 1) {
                uint8_t t = loc[j];
                for (int q = 0; q < power - 1; q++) t = gmul(t, xi_inv);
                den ^= t;
            }
        }
        if (den == 0) return 0;
        /* evaluator at xi_inv */
        uint8_t num = 0;
        for (int j = 0; j < ev_len; j++)
            num = gmul(num, xi_inv) ^ ev[j];
        uint8_t mag = gmul(xi, gdiv(num, den));
        out[p] ^= mag;
    }
    /* verify like reedsolo: recompute syndromes, must be clean */
    uint8_t chk[49];
    if (calc_synd(out, n, nsym, chk)) return 0;
    return 1;
}

/* The certify inner loop for one codeword: hard, then the ladder.
 * order: byte indices of this chunk sorted least-confident first (len 255).
 * sub_size: payload bytes after the 4-byte CRC.
 * Returns 1 and fills block[sub_size] + coded[255] on success. */
int certify_codeword(const uint8_t *chunk, const int32_t *order,
                     int ecc, int sub_size, int use_ladder,
                     uint8_t *block, uint8_t *coded) {
    init_tables();
    uint8_t dec[255];
    int ok = correct(chunk, ecc, NULL, 0, dec);
    if (!ok && use_ladder && order) {
        int top = (int)(ecc * 0.7);
        for (int n_er = 4; n_er <= top && !ok; n_er += 6)
            ok = correct(chunk, ecc, order, n_er, dec);
    }
    if (!ok) return 0;
    int msg_len = 255 - ecc;
    if (msg_len < 4 + sub_size) return 0;
    uint32_t want = (uint32_t)dec[0] | ((uint32_t)dec[1] << 8) |
                    ((uint32_t)dec[2] << 16) | ((uint32_t)dec[3] << 24);
    if (crc32_c(dec + 4, sub_size) != want) return 0;
    memcpy(block, dec + 4, sub_size);
    if (coded) memcpy(coded, dec, 255);   /* corrected codeword incl. parity */
    return 1;
}
