from score_models import print_score_segments
from stft_features import run_stage1_and_stage2, plot_stage1_diagnostics

# Try "c_major_scale", "ode_to_joy_with_harmony", or "moonlight_sonata_4m"
# bundle = run_stage1_and_stage2("recordings/aloud/odeharm.wav", model_name="ode_to_joy_with_harmony")
# bundle = run_stage1_and_stage2("recordings/aloud/c_major_asc.wav", model_name="c_major_scale")
# bundle = run_stage1_and_stage2("recordings/aloud/moonlight_sonata.wav", model_name="moonlight_sonata_4m")

bundle = run_stage1_and_stage2("recordings/wrong_notes/processed/c_scale_1h_1wrong.wav", model_name="c_major_scale")

print_score_segments(bundle.segments)
plot_stage1_diagnostics(bundle)