import numpy as np


def msg_to_enc_arr(message: str, n_rows: int, n_cols: int,
                   n_bits: int) -> np.ndarray:
    def add_ec(frame: np.ndarray) -> np.ndarray:
        row_parity = np.bitwise_xor.reduce(frame, axis=1, keepdims=True)
        frame = np.hstack([frame, row_parity])
        col_parity = np.bitwise_xor.reduce(frame, axis=0, keepdims=True)
        frame = np.vstack([frame, col_parity])
        return frame

    def make_frame(frame_bits: np.ndarray) -> np.ndarray:
        data = frame_bits.reshape(n_rows - 1, n_cols - 1)
        return add_ec(data)

    message = format(len(message), f"0{n_bits}b") + message
    chunk_size = (n_rows - 1) * (n_cols - 1)
    pad_len = (-len(message)) % chunk_size
    if pad_len:
        message += ''.join(map(str, np.random.randint(0, 2, pad_len)))
    bits = np.fromiter(message, dtype=int).reshape(-1, chunk_size)
    return np.array([make_frame(chunk) for chunk in bits])


def msg_to_enc(message: str, n_rows: int, n_cols: int, n_bits: int) -> str:
    enc_frames = msg_to_enc_arr(message, n_rows, n_cols, n_bits).flatten()
    return ''.join(map(str, enc_frames))


def enc_arr_msg_to_frames(enc_message: np.ndarray) -> np.ndarray:
    return np.pad(enc_message, ((0, 0), (1, 1), (1, 1)), mode="constant",
                  constant_values=1)


def enc_msg_to_frames(enc_message: str, n_rows: int, n_cols: int) -> np.ndarray:
    bits = np.fromiter(enc_message, dtype=int)
    frame_size = n_rows * n_cols
    num_frames = len(bits) // frame_size
    return enc_arr_msg_to_frames(bits.reshape(num_frames, n_rows, n_cols))


def msg_to_frames(message: str, n_rows: int, n_cols: int,
                  n_bits: int) -> np.ndarray:
    return enc_arr_msg_to_frames(
        msg_to_enc_arr(message, n_rows, n_cols, n_bits))


if __name__ == "__main__":
    msg = str(input("Message: "))
    frames = msg_to_frames(msg, 4, 7, 8)
    print(frames)
