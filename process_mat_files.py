import os
import scipy.io
import numpy as np
import h5py
import matplotlib.pyplot as plt

def load_mat_file(file_path, key):

    mat = scipy.io.loadmat(file_path)
    return mat[key]

def filter_control_signal(EOG, ControlSignal, exclude_value=3):

    valid_indices = np.where(ControlSignal != exclude_value)[0]
    EOG_clean = EOG[:, valid_indices]
    ControlSignal_clean = ControlSignal[valid_indices]
    return EOG_clean, valid_indices, ControlSignal_clean

def filter_into_chunks(cleaned_EOG, controlSignal_clean, cleaned_labels):
    assert cleaned_EOG.shape[1] == controlSignal_clean.shape[0]
    all_chunks = []
    all_targets = []
    j = 0
    chunk = []
    for i in range(controlSignal_clean.shape[0] - 1):
        if  controlSignal_clean[i] != controlSignal_clean[i + 1]:
            j = 0
            all_chunks.append(chunk[:256])
            all_targets.append(cleaned_labels[:,i-10])

            chunk = []
        chunk.append(cleaned_EOG[:, i])
        j += 1
    return all_chunks, all_targets


def create_windows(data, labels, window_size, step_size, threshold=256):

    num_channels, num_samples = data.shape
    num_labels, _ = labels.shape
    windows = []
    window_labels = []
    excluded_windows = 0

    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        window = data[:, start:end]
        if np.all(np.abs(window) <= threshold):
            windows.append(window)
            label_window = labels[:, end-1]
            window_labels.append(label_window)
        else:
            excluded_windows += 1

    return np.array(windows), np.array(window_labels)

def save_processed_data(windows, labels, output_file):

    with h5py.File(output_file, 'w') as h5f:
        h5f.create_dataset('windows', data=windows, compression="gzip")
        h5f.create_dataset('labels', data=labels, compression="gzip")
    print(f"Сохранено: {output_file}")

def process_all_subjects(data_dir, window_size, step_size, threshold=256):

    all_EOG = []
    all_labels = []
    total_excluded = 0

    subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    subjects_sorted = sorted(subjects, key=lambda x: int(x.replace('S', '')))

    all_EOG = np.zeros(shape=(4, 256, 4000))
    all_labels = np.zeros(shape=(2, 4000))
    for i, subject in enumerate(subjects_sorted):
        subject_path = os.path.join(data_dir, subject)
        print(f"Обработка {subject}...")

        EOG = load_mat_file(os.path.join(subject_path, 'EOG.mat'), 'EOG')  # Shape: (4, N)
        ControlSignal = load_mat_file(os.path.join(subject_path, 'ControlSignal.mat'), 'ControlSignal').flatten()  # Shape: (N,)
        Target_GA_stream = load_mat_file(os.path.join(subject_path, 'Target_GA_stream.mat'), 'Target_GA_stream')  # Shape: (2, N)

        EOG_clean, valid_indices, control_signal_clean  = filter_control_signal(EOG, ControlSignal, exclude_value=3)

        Target_GA_clean = Target_GA_stream[:, valid_indices]
        start_idx = i * 400
        end_idx = (i + 1) * 400

        chunks, labels = filter_into_chunks(EOG_clean, control_signal_clean, Target_GA_clean)

        padded_chunks = []
        for chunk in chunks:
            padding_width = 256 - len(chunk)
            if padding_width > 0:
                zer = np.zeros(shape=(256, 4))
                zer[:len(chunk), :] = chunk
                chunk_padded = np.array(zer)
            else:
                chunk_padded = chunk
            padded_chunks.append(chunk_padded)


        hihih = np.array(padded_chunks).transpose(2, 1, 0)
        print("Размер all_EOG[:, :, start_idx:end_idx]:", all_EOG[:, :, start_idx:end_idx].shape)
        print("Размер padded_chunks:", np.array(padded_chunks).transpose(2, 1, 0).shape)
        num_zeros = np.size(padded_chunks) - np.count_nonzero(padded_chunks)

        print("Количество нулей в массиве:", num_zeros)
        all_EOG[:, :, start_idx:end_idx] = np.array(padded_chunks).transpose(2, 1, 0)
        all_labels[:, start_idx:end_idx] = np.array(labels).transpose(1,0)
        x = 2
    num_zeros = np.size(all_EOG) - np.count_nonzero(all_EOG)

    print("Количество нулей в массиве:", num_zeros)
    return all_EOG, all_labels









def main():
    # Параметры
    ROOT_DIR = os.getcwd()
    DATA_DIR = os.path.join(ROOT_DIR, 'raw_data')
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'data')
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'combined_dataset.h5')

    WINDOW_SIZE = 256
    STEP_SIZE = 256
    THRESHOLD = 256

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    windowed_EOG, windowed_labels = process_all_subjects(DATA_DIR, WINDOW_SIZE, STEP_SIZE, THRESHOLD)

    save_processed_data(windowed_EOG, windowed_labels, OUTPUT_FILE)


def visualize_samples(windows, labels, num_samples=5):

    num_channels, window_size = windows.shape[1], windows.shape[2]
    for i in range(min(num_samples, len(windows))):
        plt.figure(figsize=(12, 8))
        for channel in range(num_channels):
            plt.subplot(num_channels, 1, channel+1)
            plt.plot(windows[i, channel, :], label=f'Канал {channel+1}')
            plt.legend()
        plt.suptitle(f'Метки: Горизонтальный угол = {labels[i, 0]:.2f}, Вертикальный угол = {labels[i, 1]:.2f}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    main()
    #
    # processed_file = os.path.join(os.getcwd(), 'data', 'combined_dataset.h5')
    # #
    # with h5py.File(processed_file, 'r') as h5f:
    #      windows = np.array(h5f['windows'])
    #      labels = np.array(h5f['labels'])
    #
    # # Визуализация первых 5 примеров
    # visualize_samples(windows, labels, num_samples=5)
