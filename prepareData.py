
import numpy as np
import os
import pandas as pd
import configparser

def load_time_index(files, data_config, num_steps, time_slice_size):
    """Read real timestamps, or construct them from an explicit start time."""
    if 'timestamps' in files:
        values = files['timestamps']
        if np.issubdtype(values.dtype, np.number):
            # Numeric timestamps must be Unix seconds, not a traffic feature.
            if values.min() < 946684800 or values.max() > 4102444800:
                raise ValueError("Numeric timestamps must be Unix seconds (2000-2100).")
            times = pd.DatetimeIndex(pd.to_datetime(values, unit='s'))
        else:
            times = pd.DatetimeIndex(pd.to_datetime(values))
    elif data_config.get('start_datetime', fallback='').strip():
        start = pd.Timestamp(data_config['start_datetime'])
        times = pd.date_range(start, periods=num_steps, freq=f'{time_slice_size}min')
    else:
        raise ValueError(
            "Traffic measurements are not timestamps. Provide a 'timestamps' "
            "array in the raw NPZ or [Data] start_datetime in the config."
        )
    if len(times) != num_steps or times.hasnans:
        raise ValueError("Timestamp count must match the data time axis with no missing values.")
    interval = np.diff(times.asi8)
    expected = time_slice_size * 60 * 1_000_000_000
    if len(interval) and not np.all(interval == expected):
        raise ValueError(f"Timestamps must be regular {time_slice_size}-minute intervals.")
    return times

def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = np.zeros(shape = (num_sample, num_his, dims))
    y = np.zeros(shape = (num_sample, num_pred, dims))
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y

def seq2instance_plus(data, num_his, num_pred):
    num_step = data.shape[0]
    num_sample = num_step - num_his - num_pred + 1
    x = []
    y = []
    for i in range(num_sample):
        x.append(data[i: i + num_his])
        y.append(data[i + num_his: i + num_his + num_pred, :, :1])
    x = np.array(x)
    y = np.array(y)
    return x, y
def prepare_dataset(config_path):
    print(f"Reading configuration file: {config_path}")
    config = configparser.ConfigParser()
    config.read(config_path)
    data_config = config['Data']
    training_config = config['Training'] if config.has_section('Training') else config['Model']

    time_slice_size = int(data_config['time_slice_size'])
    train_ratio = float(data_config['train_ratio'])
    val_ratio = float(data_config['val_ratio'])
    test_ratio = float(data_config['test_ratio'])
    num_his = int(training_config.get('num_his', training_config.get('in_steps')))
    num_pred = int(training_config.get('num_pred', training_config.get('out_steps')))
    num_of_vertices = int(data_config['num_of_vertices'])

    data_file = data_config['data_file']
    files = np.load(data_file, allow_pickle=False)
    data=files['data']
    print(f"Dataset shape: {data.shape}")
    if data.shape[1] != num_of_vertices:
        raise ValueError("Configured num_of_vertices does not match the raw data.")
    times = load_time_index(files, data_config, data.shape[0], time_slice_size)

    slices = data.shape[0]
    train_slices = int(slices * train_ratio)
    val_slices = int(slices * val_ratio)
    test_slices = slices - train_slices - val_slices

    train_set = data[:train_slices]
    val_set = data[train_slices:train_slices + val_slices]
    test_set = data[-test_slices:]

    sets = {'train': train_set, 'val': val_set, 'test': test_set}
    time_sets = {'train': times[:train_slices],
                 'val': times[train_slices:train_slices + val_slices],
                 'test': times[-test_slices:]}
    xy = {}
    te = {}
    origin_times = {}
    for set_name in sets.keys():
        data_set = sets[set_name]
        if len(data_set) < num_his + num_pred:
            raise ValueError(
                f"{set_name} has {len(data_set)} time steps; at least "
                f"{num_his + num_pred} are required for one sample."
            )
        X, Y = seq2instance_plus(data_set[..., :1].astype("float64"), num_his, num_pred)
        xy[set_name] = [X, Y]

        time = time_sets[set_name]
        dayofweek = np.reshape(time.weekday, (-1, 1))
        timeofday = (time.hour * 3600 + time.minute * 60 + time.second) // (time_slice_size * 60)
        timeofday = np.reshape(timeofday, (-1, 1))
        time = np.concatenate((dayofweek, timeofday), -1)
        time = seq2instance(time, num_his, num_pred)
        te[set_name] = np.concatenate(time, 1).astype(np.int32)
        origin_times[set_name] = (time_sets[set_name][
            num_his - 1:num_his - 1 + len(X)].asi8 // 1_000_000_000)

    x_trains, y_trains = xy['train'][0], xy['train'][1]
    x_vals, y_vals = xy['val'][0], xy['val'][1]
    x_tests, y_tests = xy['test'][0], xy['test'][1]
    trainTEs = te['train']
    valTEs = te['val']
    testTEs = te['test']

    print(f"train: {x_trains.shape}, {y_trains.shape}")
    print(f"val: {x_vals.shape}, {y_vals.shape}")
    print(f"test: {x_tests.shape}, {y_tests.shape}")
    print(f"trainTE: {trainTEs.shape}")
    print(f"valTE: {valTEs.shape}")
    print(f"testTE: {testTEs.shape}")

    output_dir = data_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    output_filename = data_config.get(
        'output_filename', f"samples_{num_his}_{num_pred}_{time_slice_size}.npz")
    output_path = os.path.join(output_dir, output_filename)
    print(f"Saving samples to {output_path}")
    np.savez_compressed(
        output_path,
        train_x=x_trains, train_target=y_trains,
        val_x=x_vals, val_target=y_vals,
        test_x=x_tests, test_target=y_tests,
        trainTE=trainTEs, valTE=valTEs, testTE=testTEs,
        train_origin_time=origin_times['train'],
        val_origin_time=origin_times['val'],
        test_origin_time=origin_times['test']
    )

if __name__ == "__main__":
    config_files = [
        '.../conf/PEMSD3_1dim_12.conf',
        '.../conf/PEMSD4_1dim_12.conf',
        '.../conf/PEMSD7_1dim_12.conf',
        '.../conf/PEMSD8_1dim_12.conf',
    ]

    for config_file in config_files:
        prepare_dataset(config_file)

##########################################################

# Prepare dataset separately for each configuration file
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument(

#         'config', nargs='?', default='... /conf/PEMSD7_1dim_12.conf',
#         help='Dataset configuration file (defaults to the original project path)')
#     prepare_dataset(parser.parse_args().config)
