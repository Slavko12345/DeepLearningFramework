#ifndef __globals__
#define __globals__

const bool NEW_ERROR_FUNCTION = 0;
const bool NORMALIZE_WEIGHTS = 0;
const double NORMALIZATION_RADIUS = 1.0;
const double RATE_DECAY = 0.97;
const double MAX_ABS_RANDOM_WEIGHTS = 0.01;
const bool LAPTOP = 1;
const bool LOCAL_DATA = 1;
const bool ADAPT_PROCESSORS_LOAD = 1;
const bool FOCUSED_TRAINING = 0;
const double FOCUSED_PROBABILITY_THRESHOLD = 0.5;
const bool PARALLEL_ARCHITECTURE = 1;
const int INPUT_DEPTH = 0;
const bool READ_TO_VECT = 0;

const bool DATA_AUGMENTATION = 0;
const double AUGMENTATION_SIZE = 0.2;
const double CORNER_PROBABILITY = 0.5;

const bool DROP_DATA_AUGMENTATION = 1;
const bool INPUT_UNCHANGED_PROBABILITY = 0.0625;
const bool INPUT_DROP_RATE = 0.0625;
const int INPUT_LEN = 3 * 32 * 32;

const bool INPUT_NORMALIZATION = 0;
const bool LAYERWISE_NORMALIZATION = 0;
const bool APPEND_INPUT_STATISTICS = 0;
const bool INPUT_NEG_POS = 1;
#define NET_WEIGHTS_FILE "data/w_1.txt"
#define LOG_FILE "data/log_1.txt"
#define STRUCTERED_WEIGHTS "data/str_0.txt"

#endif
