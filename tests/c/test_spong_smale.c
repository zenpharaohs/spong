#include "spong/spong_smale.h"

#include <assert.h>

static spong_rational_input q(const char *numerator, const char *denominator) {
    spong_rational_input value = {numerator, denominator};
    return value;
}

int main(void) {
    /* Synthetic scalar chart: A=1, B=0, N=1 gives exactly dy/db=2.
     * N is deliberately independent here so the C ABI test can exercise the
     * verifier without constructing a full SPONG model. */
    spong_rational_input A[] = {{"1", "1"}};
    spong_rational_input B[] = {{"0", "1"}};
    spong_rational_input N[] = {{"1", "1"}};
    spong_exact_loss_pencil pencil = {A, 1, B, 1, N, 1, {"0", "1"}};
    spong_b_parameter_centre centres[4];
    centres[0].level = q("1", "1");
    centres[0].b = q("0", "1");
    centres[0].y = q("1", "1");
    centres[1].level = q("25", "16");
    centres[1].b = q("1", "8");
    centres[1].y = q("5", "4");
    centres[2].level = q("9", "4");
    centres[2].b = q("1", "4");
    centres[2].y = q("3", "2");
    centres[3].level = q("49", "16");
    centres[3].b = q("3", "8");
    centres[3].y = q("7", "4");
    spong_rational_input initial_lo = {"1099511627775", "1099511627776"};
    spong_rational_input initial_hi = {"1099511627777", "1099511627776"};
    spong_rational_input target = {"121", "100"};
    spong_b_parameter_policy policy = {8, 10, 10, 4096, 16384, 192, 0};
    spong_b_parameter_result result;
    assert(spong_b_parameter_handoff_decimal(&pencil, centres, 4, &initial_lo,
                                             &initial_hi, &target, NULL, &policy,
                                             &result) == 0);
    assert(result.status == SPONG_SMALE_OK);
    assert(result.tube_validated == 1);
    assert(result.projection_validated == 1);
    assert(result.b_direction == 1);
    assert(result.work.slabs_accepted >= 1);
    assert(result.target_b.lower.numerator != NULL);
    assert(result.target_y.upper.denominator != NULL);
    spong_b_parameter_result_destroy(&result);

    /* A production request rejects the synthetic, inconsistent N. */
    policy.verify_n_identity = 1;
    assert(spong_b_parameter_handoff_decimal(&pencil, centres, 4, &initial_lo,
                                             &initial_hi, &target, NULL, &policy,
                                             &result) != 0);
    assert(result.status == SPONG_SMALE_MODEL_IDENTITY_FAILURE);
    assert(result.primary_reason == SPONG_SMALE_REASON_N_IDENTITY);
    spong_b_parameter_result_destroy(&result);

    /* Exact fixed-sheet tube on A=1-b^2, B=1, N=-2b.  The centreline
     * b=0 is invariant and y=sqrt(1+level) is rational at these levels. */
    spong_rational_input sheet_A[] = {{"1", "1"}, {"0", "1"}, {"-1", "1"}};
    spong_rational_input sheet_B[] = {{"1", "1"}};
    spong_rational_input sheet_N[] = {{"0", "1"}, {"-2", "1"}};
    spong_exact_loss_pencil sheet_pencil = {sheet_A, 3, sheet_B,   1,
                                            sheet_N, 2, {"0", "1"}};
    spong_b_parameter_centre sheet_centres[3];
    sheet_centres[0].level = q("0", "1");
    sheet_centres[0].b = q("0", "1");
    sheet_centres[0].y = q("1", "1");
    sheet_centres[1].level = q("5", "4");
    sheet_centres[1].b = q("0", "1");
    sheet_centres[1].y = q("3", "2");
    sheet_centres[2].level = q("3", "1");
    sheet_centres[2].b = q("0", "1");
    sheet_centres[2].y = q("2", "1");
    spong_rational_input sheet_b_lo = {"-1", "1048576"};
    spong_rational_input sheet_b_hi = {"1", "1048576"};
    spong_rational_input sheet_y_lo = {"1048575", "1048576"};
    spong_rational_input sheet_y_hi = {"1048577", "1048576"};
    spong_sheet_tube_policy sheet_policy = {8, 10, 10, 4096, 16384, 192, 192, 1};
    spong_sheet_tube_result sheet_result;
    assert(spong_sheet_flow_tube_decimal(&sheet_pencil, sheet_centres, 3, &sheet_b_lo,
                                         &sheet_b_hi, &sheet_y_lo, &sheet_y_hi, 1, NULL,
                                         &sheet_policy, &sheet_result) == 0);
    assert(sheet_result.status == SPONG_SMALE_OK);
    assert(sheet_result.projection_validated == 1);
    assert(sheet_result.tube_validated == 1);
    assert(sheet_result.level_direction == 1);
    assert(sheet_result.work.slabs_accepted == 2);
    assert(sheet_result.terminal_b.lower.numerator != NULL);
    assert(sheet_result.terminal_y.upper.denominator != NULL);
    spong_sheet_tube_result_destroy(&sheet_result);

    /* Exact local launch: A=1-b^2, B=1 has an N-saddle at (a,b)=(1,0).
     * In the identity eigenframe its unstable branch is exactly b=0. */
    spong_rational_input local_A[] = {{"1", "1"}, {"0", "1"}, {"-1", "1"}};
    spong_rational_input local_B[] = {{"1", "1"}};
    spong_rational_input local_N[] = {{"0", "1"}, {"-2", "1"}};
    spong_exact_loss_pencil local_pencil = {local_A, 3, local_B,   1,
                                            local_N, 2, {"0", "1"}};
    spong_local_launch_request launch = {0};
    launch.critical_source = SPONG_CRITICAL_ROOT_N;
    launch.orientation = 1;
    launch.critical_b.lower = q("0", "1");
    launch.critical_b.upper = q("0", "1");
    launch.frame[0] = q("1", "1");
    launch.frame[1] = q("0", "1");
    launch.frame[2] = q("0", "1");
    launch.frame[3] = q("1", "1");
    for (int i = 0; i < 6; ++i)
        launch.selected_map[i] = q("0", "1");
    launch.departing_eigenvalue = q("2", "1");
    launch.desired_reach = q("1", "16");
    launch.require_frobenius = 1;
    spong_local_launch_policy launch_policy = {24, 96, 80, 40, 1, 16384, 1};
    spong_local_launch_result launch_result;
    assert(spong_local_launch_decimal(&local_pencil, &launch, &launch_policy,
                                      &launch_result) == 0);
    assert(launch_result.status == SPONG_SMALE_OK);
    assert(launch_result.validated == 1);
    assert(launch_result.cone_power == 2);
    assert(launch_result.time_direction == 1);
    assert(launch_result.section_y.lower.numerator != NULL);
    spong_local_launch_result_destroy(&launch_result);

    /* Swap the exact eigenframe so u is the physical b direction.  The
     * Frobenius cone then has a regular native fixed-b section. */
    spong_local_launch_request b_launch = launch;
    b_launch.frame[0] = q("0", "1");
    b_launch.frame[1] = q("1", "1");
    b_launch.frame[2] = q("1", "1");
    b_launch.frame[3] = q("0", "1");
    b_launch.departing_eigenvalue = q("-2", "1");
    assert(spong_local_launch_decimal(&local_pencil, &b_launch, &launch_policy,
                                      &launch_result) == 0);
    assert(launch_result.validated == 1);
    assert(launch_result.b_section_validated == 1);
    assert(launch_result.b_direction == 1);
    assert(launch_result.b_section_b.numerator != NULL);
    assert(launch_result.b_section_y.lower.denominator != NULL);
    spong_local_launch_result_destroy(&launch_result);

    launch.critical_b.lower = q("1", "2");
    launch.critical_b.upper = q("1", "2");
    assert(spong_local_launch_decimal(&local_pencil, &launch, &launch_policy,
                                      &launch_result) != 0);
    assert(launch_result.primary_reason == SPONG_SMALE_REASON_ROOT_INTERVAL);
    spong_local_launch_result_destroy(&launch_result);
    return 0;
}
