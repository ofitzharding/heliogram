/* crs2.c — EXACT transliteration of reedsolo.py's decoder (prim 0x11d,
 * generator 2, fcr 0), plus the certify() erasure ladder and CRC32 gate.
 *
 * Unlike the retired crs.c (ported from memory, failed its differential),
 * every function here mirrors its reedsolo namesake line by line, MSB-first
 * polynomial convention included: rs_calc_syndromes ([0] prepend and all),
 * rs_forney_syndromes, rs_find_error_locator (synd_shift, Rule B),
 * rs_find_errata_locator, rs_find_error_evaluator, rs_correct_errata,
 * rs_correct_msg. Differential-tested against reedsolo before wiring.
 *
 * Build: cc -O2 -shared -o crs2.dylib crs2.c
 */
#include <stdint.h>
#include <string.h>

#define FC 255            /* field_charac */

static uint8_t EXP[512];
static int LOG[256];
static uint32_t CRC_T[256];
static int INIT = 0;

static void init_tables(void) {
    if (INIT) return;
    int x = 1;
    for (int i = 0; i < FC; i++) {
        EXP[i] = (uint8_t)x;
        LOG[x] = i;
        x <<= 1;
        if (x & 0x100) x ^= 0x11d;
    }
    for (int i = FC; i < 512; i++) EXP[i] = EXP[i - FC];
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        CRC_T[n] = c;
    }
    INIT = 1;
}

static inline uint8_t gmul(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) return 0;
    return EXP[LOG[a] + LOG[b]];
}
static inline uint8_t gdiv(uint8_t a, uint8_t b) {
    if (a == 0) return 0;
    return EXP[(LOG[a] + FC - LOG[b]) % FC];
}
static inline uint8_t ginv(uint8_t x) { return EXP[FC - LOG[x]]; }
static inline uint8_t gpow_g(int power) {          /* generator=2, log=1 */
    int e = power % FC;
    if (e < 0) e += FC;
    return EXP[e];
}
static inline uint8_t gpow_x(uint8_t x, int power) {
    long e = ((long)LOG[x] * power) % FC;
    if (e < 0) e += FC;
    return EXP[e];
}

/* MSB-first polynomial eval, exactly gf_poly_eval */
static uint8_t poly_eval(const uint8_t *p, int lp, uint8_t x) {
    uint8_t y = p[0];
    for (int i = 1; i < lp; i++) y = gmul(y, x) ^ p[i];
    return y;
}

/* out = a * b (MSB-first), lengths la+lb-1 */
static int poly_mul(const uint8_t *a, int la, const uint8_t *b, int lb,
                    uint8_t *out) {
    int lo = la + lb - 1;
    memset(out, 0, lo);
    for (int i = 0; i < la; i++)
        for (int j = 0; j < lb; j++)
            out[i + j] ^= gmul(a[i], b[j]);
    return lo;
}

/* rs_calc_syndromes with the [0] prepend: synd[0]=0, synd[1+i]=eval(alpha^i).
 * Returns max value. */
static int calc_synd(const uint8_t *msg, int n, int nsym, uint8_t *synd) {
    int mx = 0;
    synd[0] = 0;
    for (int i = 0; i < nsym; i++) {
        uint8_t v = poly_eval(msg, n, gpow_g(i));
        synd[1 + i] = v;
        if (v > mx) mx = v;
    }
    return mx;
}

/* rs_correct_msg. Returns 0 on success (msg corrected in place), else -1. */
static int correct_msg(uint8_t *msg, int n, int nsym,
                       const int32_t *erase_pos, int n_erase) {
    uint8_t synd[64], fsynd[64];
    uint8_t err_loc[128], old_loc[128], tmp[128];
    if (n_erase > nsym) return -1;
    for (int i = 0; i < n_erase; i++) msg[erase_pos[i]] = 0;
    if (calc_synd(msg, n, nsym, synd) == 0) return 0;   /* clean */

    /* rs_forney_syndromes: fsynd = synd[1:], fold each erasure */
    int lf = nsym;
    for (int i = 0; i < lf; i++) fsynd[i] = synd[1 + i];
    for (int i = 0; i < n_erase; i++) {
        uint8_t x = gpow_g(n - 1 - erase_pos[i]);
        for (int j = 0; j < lf - 1; j++)
            fsynd[j] = gmul(fsynd[j], x) ^ fsynd[j + 1];
    }

    /* rs_find_error_locator(fsynd, nsym, erase_count): synd_shift = 0
     * (len(fsynd) == nsym), K = i, loop nsym - erase_count iterations */
    int le = 1, lo_ = 1;
    err_loc[0] = 1; old_loc[0] = 1;
    for (int i = 0; i < nsym - n_erase; i++) {
        int K = i;
        uint8_t delta = fsynd[K];
        for (int j = 1; j < le; j++)
            delta ^= gmul(err_loc[le - 1 - j], fsynd[K - j]);
        old_loc[lo_++] = 0;                     /* old_loc += [0] */
        if (delta != 0) {
            if (lo_ > le) {
                /* new_loc = scale(old_loc, delta) */
                for (int j = 0; j < lo_; j++) tmp[j] = gmul(old_loc[j], delta);
                /* old_loc = scale(err_loc, inv(delta)) */
                uint8_t di = ginv(delta);
                for (int j = 0; j < le; j++) old_loc[j] = gmul(err_loc[j], di);
                int t = lo_; lo_ = le; le = t;
                memcpy(err_loc, tmp, le);
            }
            /* err_loc = add(err_loc, scale(old_loc, delta)); MSB-first add
             * aligns the tails */
            int lm = le > lo_ ? le : lo_;
            uint8_t sum[128];
            memset(sum, 0, lm);
            for (int j = 0; j < le; j++) sum[lm - le + j] ^= err_loc[j];
            for (int j = 0; j < lo_; j++)
                sum[lm - lo_ + j] ^= gmul(old_loc[j], delta);
            memcpy(err_loc, sum, lm); le = lm;
        }
    }
    int lead = 0;
    while (lead < le && err_loc[lead] == 0) lead++;
    le -= lead;
    memmove(err_loc, err_loc + lead, le);
    int errs = le - 1;
    if ((errs - n_erase) * 2 + n_erase > nsym) return -1;

    /* rs_find_errors: Chien over reversed err_loc */
    uint8_t rev[128];
    for (int i = 0; i < le; i++) rev[i] = err_loc[le - 1 - i];
    int err_pos[128], n_err = 0;
    for (int i = 0; i < n; i++)
        if (poly_eval(rev, le, gpow_g(i)) == 0) {
            if (n_err >= 128) return -1;
            err_pos[n_err++] = n - 1 - i;
        }
    if (n_err != errs) return -1;

    /* errata positions = erase_pos + err_pos */
    int pos[160], np = 0;
    for (int i = 0; i < n_erase; i++) pos[np++] = erase_pos[i];
    for (int i = 0; i < n_err; i++) pos[np++] = err_pos[i];

    /* rs_correct_errata(msg, synd, pos):
     * coef_pos[i] = n-1-pos[i]; errata locator from coef_pos */
    int coef[160];
    for (int i = 0; i < np; i++) coef[i] = n - 1 - pos[i];
    uint8_t eloc[200]; int lel = 1; eloc[0] = 1;
    for (int i = 0; i < np; i++) {
        /* term = [gpow(g, coef), 1]  (p*x + 1, MSB-first) */
        uint8_t term[2] = { gpow_g(coef[i]), 1 };
        uint8_t prod[220];
        int lp = poly_mul(eloc, lel, term, 2, prod);
        memcpy(eloc, prod, lp); lel = lp;
    }
    /* err_eval = find_error_evaluator(synd[::-1], eloc, lel-1)[::-1]
     * = last (lel) coeffs of (synd_rev * eloc), then reversed */
    uint8_t srev[64];
    int ls = nsym + 1;
    for (int i = 0; i < ls; i++) srev[i] = synd[ls - 1 - i];
    uint8_t prod2[300];
    int lp2 = poly_mul(srev, ls, eloc, lel, prod2);
    int lev = lel;                       /* nsym_param+1 = (lel-1)+1 */
    uint8_t eval_[220];                  /* already re-reversed */
    for (int i = 0; i < lev; i++) eval_[i] = prod2[lp2 - 1 - i];

    /* X[i] = gpow(g, -(FC - coef[i])) */
    uint8_t X[160];
    for (int i = 0; i < np; i++) X[i] = gpow_g(-(FC - coef[i]));

    uint8_t E[255];
    memset(E, 0, n);
    for (int i = 0; i < np; i++) {
        uint8_t Xi = X[i];
        uint8_t Xi_inv = ginv(Xi);
        uint8_t prime = 1;
        for (int j = 0; j < np; j++)
            if (j != i) prime = gmul(prime, 1 ^ gmul(Xi_inv, X[j]));
        if (prime == 0) return -1;
        /* y = poly_eval(err_eval[::-1], Xi_inv): reverse eval_ again */
        uint8_t evrev[220];
        for (int j = 0; j < lev; j++) evrev[j] = eval_[lev - 1 - j];
        uint8_t y = poly_eval(evrev, lev, Xi_inv);
        y = gmul(gpow_x(Xi, 1), y);          /* fcr = 0 -> Xi^(1-0) */
        E[pos[i]] = gdiv(y, prime);
    }
    for (int i = 0; i < n; i++) msg[i] ^= E[i];
    if (calc_synd(msg, n, nsym, synd) != 0) return -1;
    return 0;
}

uint32_t crc32_c(const uint8_t *buf, int len) {
    init_tables();
    uint32_t c = 0xFFFFFFFFu;
    for (int i = 0; i < len; i++)
        c = CRC_T[(c ^ buf[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

/* certify() inner loop: hard decode, then the erasure ladder
 * (n_er = 4, 10, .., <= 0.7*ecc), CRC32 gate, corrected codeword out.
 * order: 255 byte indices, least confident first (numpy argsort).
 * Returns 1 on success with block (sub_size bytes) and coded (255). */
int certify_codeword(const uint8_t *chunk, const int32_t *order,
                     int ecc, int sub_size, int use_ladder,
                     uint8_t *block, uint8_t *coded) {
    init_tables();
    uint8_t msg[255];
    int n = 255, ok = 0;
    memcpy(msg, chunk, n);
    ok = (correct_msg(msg, n, ecc, NULL, 0) == 0);
    if (!ok && use_ladder && order) {
        int top = (int)(ecc * 0.7);
        for (int n_er = 4; n_er <= top && !ok; n_er += 6) {
            memcpy(msg, chunk, n);
            ok = (correct_msg(msg, n, ecc, order, n_er) == 0);
        }
    }
    if (!ok) return 0;
    int msg_len = n - ecc;
    if (msg_len < 4 + sub_size) return 0;
    uint32_t want = (uint32_t)msg[0] | ((uint32_t)msg[1] << 8) |
                    ((uint32_t)msg[2] << 16) | ((uint32_t)msg[3] << 24);
    if (crc32_c(msg + 4, sub_size) != want) return 0;
    memcpy(block, msg + 4, sub_size);
    if (coded) memcpy(coded, msg, n);
    return 1;
}

/* ---- iOS bridge (ios/SPEC.md §4): raw RS decode for the 68-byte header.
 * The payload path goes through certify_codeword; the header is RS(68,28)
 * with an 8-phase mask and its own accept test, so it needs correct_msg
 * without the CRC32/sub-block gate. This is the only addition to the file. */
int rs_correct(uint8_t *msg, int n, int nsym,
               const int32_t *erase, int n_erase) {
    init_tables();
    return correct_msg(msg, n, nsym, erase, n_erase);
}
