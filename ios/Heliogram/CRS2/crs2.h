/* crs2.h — Swift-facing surface of the C decode core (src/crs2.c, copied
 * verbatim into CRS2/crs2.c plus the one appended rs_correct wrapper).
 * Bridging header for the Heliogram target. Nothing here is reimplemented
 * in Swift: this is the hot path and it is already differential-tested
 * against reedsolo. */
#ifndef CRS2_H
#define CRS2_H

#include <stdint.h>

/* zlib polynomial (reflected 0xEDB88320), init and final xor 0xFFFFFFFF */
uint32_t crc32_c(const uint8_t *buf, int len);

/* chunk: 255 sampled bytes.
 * order: 255 byte indices, least confident first, or NULL for hard-only.
 * use_ladder: run the erasure ladder (n_er = 4, 10, ... <= 0.7*ecc).
 * Returns 1 and fills block[sub_size] (and coded[255] when non-NULL). */
int certify_codeword(const uint8_t *chunk, const int32_t *order,
                     int ecc, int sub_size, int use_ladder,
                     uint8_t *block, uint8_t *coded);

/* raw RS decode in place; returns 0 on success. Used for the header,
 * n = 68, nsym = 40. */
int rs_correct(uint8_t *msg, int n, int nsym,
               const int32_t *erase, int n_erase);

#endif
