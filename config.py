"""Central configuration for the CL1 video generation system."""

# ── Channel Layout ────────────────────────────────────────────────────────────
TOTAL_CHANNELS = 64
DEAD_CHANNELS = [0, 4, 7, 56, 63]
ACTIVE_CHANNELS = [i for i in range(TOTAL_CHANNELS) if i not in set(DEAD_CHANNELS)]  # 59

ENCODER_CHANNEL_COUNT  = 42
SPATIAL_CHANNEL_COUNT  = 21   # first half of encoder channels — edge/spatial round
COLOR_CHANNEL_COUNT    = 21   # second half of encoder channels — color/chroma round
POS_FEEDBACK_COUNT     = 8
NEG_FEEDBACK_COUNT     = 8

ENCODER_CHANNELS         = ACTIVE_CHANNELS[:ENCODER_CHANNEL_COUNT]
SPATIAL_ENCODER_CHANNELS = ENCODER_CHANNELS[:SPATIAL_CHANNEL_COUNT]
COLOR_ENCODER_CHANNELS   = ENCODER_CHANNELS[SPATIAL_CHANNEL_COUNT:]
POS_FEEDBACK_CHANNELS    = ACTIVE_CHANNELS[ENCODER_CHANNEL_COUNT : ENCODER_CHANNEL_COUNT + POS_FEEDBACK_COUNT]
NEG_FEEDBACK_CHANNELS    = ACTIVE_CHANNELS[ENCODER_CHANNEL_COUNT + POS_FEEDBACK_COUNT : ENCODER_CHANNEL_COUNT + POS_FEEDBACK_COUNT + NEG_FEEDBACK_COUNT]

# ── Stimulation Bounds ────────────────────────────────────────────────────────
MIN_FREQ_HZ = 4.0
MAX_FREQ_HZ = 40.0
MIN_AMP_UA = 1.0
MAX_AMP_UA = 2.5

# ── Feedback Thresholds ───────────────────────────────────────────────────────
POS_SSIM_THRESHOLD = 0.6        # SSIM above this → positive feedback
NEG_SSIM_THRESHOLD = 0.4        # SSIM below this → negative feedback
POS_FEEDBACK_FREQ_HZ = 30.0     # synchronous, predictable burst frequency
NEG_FEEDBACK_FREQ_MIN_HZ = 20.0 # baseline chaos frequency (scales up with error)

# ── Dataset ───────────────────────────────────────────────────────────────────
UCF101_ROOT = "./data/train"
UCF101_CLASSES = ["IceDancing", "Biking"]
SAMPLES_PER_CLASS = 50
FRAME_SIZE = (64, 64)
NUM_CLASSES = len(UCF101_CLASSES)

# ── Model ─────────────────────────────────────────────────────────────────────
STIM_ROUNDS = 3                              # stimulation rounds per frame (full, spatial, color)
SPIKE_DIM = len(ACTIVE_CHANNELS) * STIM_ROUNDS  # 177
CLASS_EMB_DIM = 32
ENCODER_HIDDEN = 512

# ── Training ──────────────────────────────────────────────────────────────────
LR_ENCODER = 1e-4
LR_DECODER = 1e-5
REINFORCE_BASELINE_DECAY = 0.99
MAX_TRAIN_SECONDS = int(2.5 * 3600)  # 2.5 hours before mandatory rest
REST_SECONDS = int(1.0 * 3600)       # 1 hour rest

# Wait this long after stimulation before accepting spikes (artifact rejection)
SPIKE_ARTIFACT_WAIT_S = 0.010  # 10 ms
INTER_CLIP_PAUSE_S    = 1.0    # pause between video clips to let cells settle

# ── UDP ───────────────────────────────────────────────────────────────────────
CL1_DEVICE = "cl1-2544-122.corticalcloud"  # cloud device identifier
CL1_HOST = "127.1.0.1"    # neural interface address — stim is sent TO this
STIM_PORT = 12345          # port on the neural interface we send stim TO
LISTEN_HOST = "0.0.0.0"   # our local bind address for receiving spike packets
SPIKE_PORT = 12346         # our local port we bind to receive spike packets
UDP_TIMEOUT_S = 5.0        # max wait for a spike response

# ── Logging / Checkpointing ───────────────────────────────────────────────────
LOG_INTERVAL = 50
CHECKPOINT_INTERVAL = 500
CHECKPOINT_DIR = "./checkpoints"
