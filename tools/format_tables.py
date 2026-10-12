import json

t1 = json.load(open('n_runs_summary_unlimited.json'))['summary_by_round']
print("TABLE 1:")
for r in t1:
    rnd = r['round']
    n_mean = r['n_attacks']['mean']
    n_std = r['n_attacks']['std']
    asr_m = r['attack_success_rate']['mean']
    asr_s = r['attack_success_rate']['std']
    asr_ci = r['attack_success_rate']['ci95']
    det_m = r['detection_rate']['mean']
    det_s = r['detection_rate']['std']
    det_ci = r['detection_rate']['ci95']
    fpr_m = r['fpr_combined']['mean']
    fpr_s = r['fpr_combined']['std']
    corp_m = r['corpus_size']['mean']
    corp_s = r['corpus_size']['std']
    flag = '*' if r.get('sample_size_flag') else ''
    print(f"| **{rnd}** | ${n_mean:.1f} \\pm {n_std:.1f}${flag} | ${asr_m:.1f} \\pm {asr_s:.1f}$ $[\\pm {asr_ci:.1f}]$ | ${det_m:.1f} \\pm {det_s:.1f}$ $[\\pm {det_ci:.1f}]$ | ${fpr_m:.1f} \\pm {fpr_s:.1f}$ | ${corp_m:.1f} \\pm {corp_s:.1f}$ |")

print("\nTABLE 2:")
t2 = json.load(open('compare_arms_metrics.json'))['summary_by_round']
for r in t2:
    rnd = r['round']
    n_mean = r['matched_n']['mean']
    n_std = r['matched_n']['std']
    flag = '*' if r.get('sample_size_flag') else ''

    s_asr_m = r['static_arm']['asr']['mean']
    s_asr_s = r['static_arm']['asr']['std']
    s_det_m = r['static_arm']['detection_rate']['mean']
    s_det_s = r['static_arm']['detection_rate']['std']
    s_fpr_m = r['static_arm']['fpr_combined']['mean']
    s_fpr_c = r['static_arm']['fpr_combined_count']['mean']
    s_fpr_d = r['static_arm']['fpr_combined_denominator']

    a_asr_m = r['adaptive_arm']['asr']['mean']
    a_asr_s = r['adaptive_arm']['asr']['std']
    a_det_m = r['adaptive_arm']['detection_rate']['mean']
    a_det_s = r['adaptive_arm']['detection_rate']['std']
    a_fpr_m = r['adaptive_arm']['fpr_combined']['mean']
    a_fpr_c = r['adaptive_arm']['fpr_combined_count']['mean']
    a_fpr_d = r['adaptive_arm']['fpr_combined_denominator']
    a_corp_m = r['adaptive_arm']['corpus_size']['mean']
    a_corp_s = r['adaptive_arm']['corpus_size']['std']

    s_fpr_str = f"{s_fpr_m:.2f}% ({s_fpr_c:.0f}/{s_fpr_d})"
    a_fpr_str = f"**{a_fpr_m:.2f}% ({a_fpr_c:.0f}/{a_fpr_d})**"

    print(f"| **{rnd}** | ${n_mean:.1f} \\pm {n_std:.1f}${flag} | ${s_asr_m:.1f} \\pm {s_asr_s:.1f}\\%$ | ${s_det_m:.1f} \\pm {s_det_s:.1f}\\%$ | {s_fpr_str} | **${a_asr_m:.1f} \\pm {a_asr_s:.1f}\\%** | **${a_det_m:.1f} \\pm {a_det_s:.1f}\\%** | {a_fpr_str} | ${a_corp_m:.1f} \\pm {a_corp_s:.1f}$ |")
