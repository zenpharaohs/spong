#include "spong/spong_local.h"

#include <gmp.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t nu;
    size_t ns;
    mpf_t *c;
} fpoly;

static size_t index_of(const fpoly *p, size_t i, size_t j) {
    return i*p->ns+j;
}

static int fpoly_init(fpoly *p, size_t nu, size_t ns, mp_bitcnt_t precision) {
    memset(p, 0, sizeof(*p));
    if (nu == 0 || ns == 0 || nu > SIZE_MAX/ns)
        return -1;
    size_t count = nu*ns;
    p->c = (mpf_t *)calloc(count, sizeof(mpf_t));
    if (p->c == NULL)
        return -1;
    p->nu = nu;
    p->ns = ns;
    for (size_t i = 0; i < count; ++i)
        mpf_init2(p->c[i], precision);
    return 0;
}

static void fpoly_clear(fpoly *p) {
    if (p->c != NULL) {
        for (size_t i = 0; i < p->nu*p->ns; ++i)
            mpf_clear(p->c[i]);
        free(p->c);
    }
    memset(p, 0, sizeof(*p));
}

static int fpoly_mul(
        fpoly *out, const fpoly *a, const fpoly *b,
        mp_bitcnt_t precision) {
    if (a->nu > SIZE_MAX-b->nu+1 || a->ns > SIZE_MAX-b->ns+1)
        return -1;
    if (fpoly_init(out, a->nu+b->nu-1, a->ns+b->ns-1, precision) != 0)
        return -1;
    mpf_t product;
    mpf_init2(product, precision);
    for (size_t i = 0; i < a->nu; ++i) {
        for (size_t j = 0; j < a->ns; ++j) {
            if (mpf_sgn(a->c[index_of(a, i, j)]) == 0)
                continue;
            for (size_t k = 0; k < b->nu; ++k) {
                for (size_t l = 0; l < b->ns; ++l) {
                    if (mpf_sgn(b->c[index_of(b, k, l)]) == 0)
                        continue;
                    mpf_mul(
                        product, a->c[index_of(a, i, j)],
                        b->c[index_of(b, k, l)]);
                    mpf_add(
                        out->c[index_of(out, i+k, j+l)],
                        out->c[index_of(out, i+k, j+l)], product);
                }
            }
        }
    }
    mpf_clear(product);
    return 0;
}

static void fpoly_add_scaled(
        fpoly *out, const fpoly *term, const mpf_t scale, int subtract,
        mp_bitcnt_t precision) {
    mpf_t product;
    mpf_init2(product, precision);
    for (size_t i = 0; i < term->nu; ++i) {
        for (size_t j = 0; j < term->ns; ++j) {
            mpf_mul(product, term->c[index_of(term, i, j)], scale);
            if (subtract)
                mpf_sub(
                    out->c[index_of(out, i, j)],
                    out->c[index_of(out, i, j)], product);
            else
                mpf_add(
                    out->c[index_of(out, i, j)],
                    out->c[index_of(out, i, j)], product);
        }
    }
    mpf_clear(product);
}

static void fpoly_add_unscaled(fpoly *out, const fpoly *term, int subtract) {
    for (size_t i = 0; i < term->nu; ++i) {
        for (size_t j = 0; j < term->ns; ++j) {
            if (subtract)
                mpf_sub(
                    out->c[index_of(out, i, j)],
                    out->c[index_of(out, i, j)],
                    term->c[index_of(term, i, j)]);
            else
                mpf_add(
                    out->c[index_of(out, i, j)],
                    out->c[index_of(out, i, j)],
                    term->c[index_of(term, i, j)]);
        }
    }
}

static int fpoly_product_difference(
        fpoly *out, const fpoly *a, const fpoly *b,
        const fpoly *c, const fpoly *d, mp_bitcnt_t precision) {
    fpoly first = {0}, second = {0};
    if (fpoly_mul(&first, a, b, precision) != 0
            || fpoly_mul(&second, c, d, precision) != 0)
        goto failure;
    size_t nu = first.nu > second.nu ? first.nu : second.nu;
    size_t ns = first.ns > second.ns ? first.ns : second.ns;
    if (fpoly_init(out, nu, ns, precision) != 0)
        goto failure;
    fpoly_add_unscaled(out, &first, 0);
    fpoly_add_unscaled(out, &second, 1);
    fpoly_clear(&first);
    fpoly_clear(&second);
    return 0;
failure:
    fpoly_clear(&first);
    fpoly_clear(&second);
    return -1;
}

static int fpoly_set_decimal(mpf_t value, const char *text) {
    if (text == NULL)
        return -1;
    size_t length = strlen(text);
    char *normalized = (char *)malloc(length+1);
    if (normalized == NULL)
        return -1;
    memcpy(normalized, text, length+1);
    for (size_t i = 0; i < length; ++i)
        if (normalized[i] == 'e' || normalized[i] == 'E')
            normalized[i] = '@';
    int status = mpf_set_str(value, normalized, 10) == 0 ? 0 : -1;
    free(normalized);
    return status;
}

static void fpoly_array_clear(fpoly *array, size_t count) {
    if (array != NULL) {
        for (size_t i = 0; i < count; ++i)
            fpoly_clear(&array[i]);
        free(array);
    }
}

void spong_vector_polynomial_destroy(spong_vector_polynomial *polynomial) {
    if (polynomial == NULL)
        return;
    free(polynomial->coefficients);
    memset(polynomial, 0, sizeof(*polynomial));
}

int spong_poincare_pullback_decimal(
        const char *const *normal_coefficients,
        size_t normal_u_count,
        size_t normal_s_count,
        const double selected_map[6],
        const char *unstable_eigenvalue,
        const char *stable_eigenvalue,
        uint64_t precision_bits,
        spong_vector_polynomial *result) {
    if (result == NULL)
        return SPONG_EXACT_INVALID_ARGUMENT;
    memset(result, 0, sizeof(*result));
    if (normal_coefficients == NULL || normal_u_count == 0
            || normal_s_count == 0 || selected_map == NULL
            || unstable_eigenvalue == NULL || stable_eigenvalue == NULL)
        return SPONG_EXACT_INVALID_ARGUMENT;
    if (normal_u_count > SIZE_MAX/normal_s_count
            || normal_u_count*normal_s_count > SIZE_MAX/2)
        return SPONG_EXACT_INVALID_ARGUMENT;
    mp_bitcnt_t precision = (mp_bitcnt_t)(precision_bits < 64
        ? 64 : precision_bits);
    size_t normal_count = normal_u_count*normal_s_count;
    fpoly normal[2] = {{0}, {0}}, map[2] = {{0}, {0}};
    fpoly jacobian[4] = {{0}, {0}, {0}, {0}};
    fpoly fields[2] = {{0}, {0}}, pulled[2] = {{0}, {0}};
    fpoly *upowers = NULL, *spowers = NULL;
    mpf_t lambda[2];
    mpf_init2(lambda[0], precision);
    mpf_init2(lambda[1], precision);
    int status = SPONG_EXACT_INTERNAL_FAILURE;

    for (size_t component = 0; component < 2; ++component) {
        if (fpoly_init(
                &normal[component], normal_u_count, normal_s_count,
                precision) != 0)
            goto allocation_failure;
        for (size_t i = 0; i < normal_count; ++i) {
            if (fpoly_set_decimal(
                    normal[component].c[i],
                    normal_coefficients[component*normal_count+i]) != 0)
                goto parse_failure;
        }
        if (fpoly_init(&map[component], 3, 3, precision) != 0)
            goto allocation_failure;
    }
    mpf_set_ui(map[0].c[index_of(&map[0], 1, 0)], 1);
    mpf_set_ui(map[1].c[index_of(&map[1], 0, 1)], 1);
    for (size_t component = 0; component < 2; ++component) {
        mpf_set_d(map[component].c[index_of(&map[component], 2, 0)],
                  selected_map[3*component]);
        mpf_set_d(map[component].c[index_of(&map[component], 1, 1)],
                  selected_map[3*component+1]);
        mpf_set_d(map[component].c[index_of(&map[component], 0, 2)],
                  selected_map[3*component+2]);
    }
    if (fpoly_set_decimal(lambda[0], unstable_eigenvalue) != 0
            || fpoly_set_decimal(lambda[1], stable_eigenvalue) != 0)
        goto parse_failure;

    upowers = (fpoly *)calloc(normal_u_count, sizeof(fpoly));
    spowers = (fpoly *)calloc(normal_s_count, sizeof(fpoly));
    if (upowers == NULL || spowers == NULL)
        goto allocation_failure;
    if (fpoly_init(&upowers[0], 1, 1, precision) != 0
            || fpoly_init(&spowers[0], 1, 1, precision) != 0)
        goto allocation_failure;
    mpf_set_ui(upowers[0].c[0], 1);
    mpf_set_ui(spowers[0].c[0], 1);
    for (size_t i = 1; i < normal_u_count; ++i) {
        if (fpoly_mul(&upowers[i], &upowers[i-1], &map[0], precision) != 0)
            goto allocation_failure;
    }
    for (size_t j = 1; j < normal_s_count; ++j) {
        if (fpoly_mul(&spowers[j], &spowers[j-1], &map[1], precision) != 0)
            goto allocation_failure;
    }
    size_t field_nu = upowers[normal_u_count-1].nu
        + spowers[normal_s_count-1].nu-1;
    size_t field_ns = upowers[normal_u_count-1].ns
        + spowers[normal_s_count-1].ns-1;
    for (size_t component = 0; component < 2; ++component) {
        if (fpoly_init(&fields[component], field_nu, field_ns, precision) != 0)
            goto allocation_failure;
        for (size_t i = 0; i < normal_u_count; ++i) {
            for (size_t j = 0; j < normal_s_count; ++j) {
                size_t coefficient = index_of(&normal[component], i, j);
                if (mpf_sgn(normal[component].c[coefficient]) == 0)
                    continue;
                fpoly term = {0};
                if (fpoly_mul(&term, &upowers[i], &spowers[j], precision) != 0)
                    goto allocation_failure;
                fpoly_add_scaled(
                    &fields[component], &term,
                    normal[component].c[coefficient], 0, precision);
                fpoly_clear(&term);
            }
        }
    }

    for (size_t i = 0; i < 4; ++i) {
        if (fpoly_init(&jacobian[i], 2, 2, precision) != 0)
            goto allocation_failure;
    }
    mpf_set_ui(jacobian[0].c[index_of(&jacobian[0], 0, 0)], 1);
    mpf_set_d(jacobian[0].c[index_of(&jacobian[0], 1, 0)], 2*selected_map[0]);
    mpf_set_d(jacobian[0].c[index_of(&jacobian[0], 0, 1)], selected_map[1]);
    mpf_set_d(jacobian[1].c[index_of(&jacobian[1], 1, 0)], selected_map[1]);
    mpf_set_d(jacobian[1].c[index_of(&jacobian[1], 0, 1)], 2*selected_map[2]);
    mpf_set_d(jacobian[2].c[index_of(&jacobian[2], 1, 0)], 2*selected_map[3]);
    mpf_set_d(jacobian[2].c[index_of(&jacobian[2], 0, 1)], selected_map[4]);
    mpf_set_ui(jacobian[3].c[index_of(&jacobian[3], 0, 0)], 1);
    mpf_set_d(jacobian[3].c[index_of(&jacobian[3], 1, 0)], selected_map[4]);
    mpf_set_d(jacobian[3].c[index_of(&jacobian[3], 0, 1)], 2*selected_map[5]);

    if (fpoly_product_difference(
            &pulled[0], &jacobian[3], &fields[0],
            &jacobian[1], &fields[1], precision) != 0
            || fpoly_product_difference(
                &pulled[1], &jacobian[0], &fields[1],
                &jacobian[2], &fields[0], precision) != 0)
        goto allocation_failure;

    mpf_set_ui(pulled[0].c[index_of(&pulled[0], 0, 0)], 0);
    mpf_set(pulled[0].c[index_of(&pulled[0], 1, 0)], lambda[0]);
    mpf_set_ui(pulled[0].c[index_of(&pulled[0], 0, 1)], 0);
    mpf_set_ui(pulled[1].c[index_of(&pulled[1], 0, 0)], 0);
    mpf_set_ui(pulled[1].c[index_of(&pulled[1], 1, 0)], 0);
    mpf_set(pulled[1].c[index_of(&pulled[1], 0, 1)], lambda[1]);

    size_t out_nu = pulled[0].nu > pulled[1].nu
        ? pulled[0].nu : pulled[1].nu;
    size_t out_ns = pulled[0].ns > pulled[1].ns
        ? pulled[0].ns : pulled[1].ns;
    while (out_nu > 2) {
        int nonzero = 0;
        for (size_t component = 0; component < 2 && !nonzero; ++component)
            for (size_t j = 0; j < out_ns; ++j)
                if (out_nu-1 < pulled[component].nu
                        && j < pulled[component].ns
                        && mpf_sgn(pulled[component].c[
                            index_of(&pulled[component], out_nu-1, j)]) != 0)
                    nonzero = 1;
        if (nonzero)
            break;
        --out_nu;
    }
    while (out_ns > 2) {
        int nonzero = 0;
        for (size_t component = 0; component < 2 && !nonzero; ++component)
            for (size_t i = 0; i < out_nu; ++i)
                if (i < pulled[component].nu
                        && out_ns-1 < pulled[component].ns
                        && mpf_sgn(pulled[component].c[
                            index_of(&pulled[component], i, out_ns-1)]) != 0)
                    nonzero = 1;
        if (nonzero)
            break;
        --out_ns;
    }
    if (out_nu > SIZE_MAX/out_ns || out_nu*out_ns > SIZE_MAX/2
            || 2*out_nu*out_ns > SIZE_MAX/sizeof(double))
        goto allocation_failure;
    result->coefficients = (double *)calloc(
        2*out_nu*out_ns, sizeof(double));
    if (result->coefficients == NULL)
        goto allocation_failure;
    result->u_count = out_nu;
    result->s_count = out_ns;
    for (size_t component = 0; component < 2; ++component) {
        for (size_t i = 0; i < out_nu; ++i) {
            for (size_t j = 0; j < out_ns; ++j) {
                if (i < pulled[component].nu && j < pulled[component].ns)
                    result->coefficients[
                        (component*out_nu+i)*out_ns+j] = mpf_get_d(
                            pulled[component].c[
                                index_of(&pulled[component], i, j)]);
            }
        }
    }

    for (size_t i = 0; i < 2; ++i) {
        fpoly_clear(&normal[i]);
        fpoly_clear(&map[i]);
        fpoly_clear(&fields[i]);
        fpoly_clear(&pulled[i]);
    }
    for (size_t i = 0; i < 4; ++i)
        fpoly_clear(&jacobian[i]);
    fpoly_array_clear(upowers, normal_u_count);
    fpoly_array_clear(spowers, normal_s_count);
    mpf_clear(lambda[0]);
    mpf_clear(lambda[1]);
    return SPONG_EXACT_OK;

parse_failure:
    status = SPONG_EXACT_PARSE_FAILURE;
    goto failure;
allocation_failure:
    status = SPONG_EXACT_ALLOCATION_FAILURE;
failure:
    for (size_t i = 0; i < 2; ++i) {
        fpoly_clear(&normal[i]);
        fpoly_clear(&map[i]);
        fpoly_clear(&fields[i]);
        fpoly_clear(&pulled[i]);
    }
    for (size_t i = 0; i < 4; ++i)
        fpoly_clear(&jacobian[i]);
    fpoly_array_clear(upowers, normal_u_count);
    fpoly_array_clear(spowers, normal_s_count);
    mpf_clear(lambda[0]);
    mpf_clear(lambda[1]);
    spong_vector_polynomial_destroy(result);
    return status;
}
