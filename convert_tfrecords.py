import tensorflow as tf
import numpy as np
import os


RAW_DIR = "raw"
TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(folder, exist_ok=True)


def _parse_function(proto):
    keys_to_features = {
        'global_view': tf.io.FixedLenFeature([2001], tf.float32),
        'local_view': tf.io.FixedLenFeature([201], tf.float32),
        'av_training_set': tf.io.FixedLenFeature([], tf.string),
        'kepid': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    return parsed_features['kepid'], parsed_features['global_view'], parsed_features['local_view'], parsed_features['av_training_set']

tfrecord_files = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if not f.startswith(".")]

total_pc, total_npc = 0, 0
kepler_id_counts = {}

for file in tfrecord_files:
    print(f"File: {file}")
    dataset = tf.data.TFRecordDataset(file).map(_parse_function)

    for kepid, global_view, local_view, label in dataset:
        kepid = int(kepid.numpy())
        label = label.numpy().decode("utf-8")
        global_view = global_view.numpy()
        local_view = local_view.numpy()

        y_label = 1 if label == "PC" else 0

        combined_lightcurve = np.concatenate((global_view, local_view))  # 2001 + 201 = 2202

        if kepid in kepler_id_counts:
            kepler_id_counts[kepid] += 1
        else:
            kepler_id_counts[kepid] = 1
        
        unique_filename = f"{kepid}_{kepler_id_counts[kepid]}.npy"

        if "train" in file:
            target_dir = TRAIN_DIR
        elif "val" in file:
            target_dir = VAL_DIR
        elif "test" in file:
            target_dir = TEST_DIR
        else:
            continue
        
        final_save_path = os.path.join(target_dir, unique_filename)
        np.save(final_save_path, {"lightcurve": combined_lightcurve, "label": y_label, "kepler_id": kepid})
        
        print(f"Saved: {final_save_path}")

        if y_label == 1:
            total_pc += 1
        else:
            total_npc += 1

print(f"Total: PC = {total_pc}, NPC = {total_npc}")
