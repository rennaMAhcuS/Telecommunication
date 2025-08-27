import logging as log
import time
from time import sleep

import cv2
import numpy as np

from settings import CAM_NO, N_BITS, N_ROWS, N_COLS, TIMEIN_RECEIVER, \
    TIMEOUT_RECEIVER, COLOR_AREA

log.basicConfig(level=log.DEBUG, format="[%(levelname)s] %(message)s")

# CV capture
cap = cv2.VideoCapture(CAM_NO)


def detect_color(color):
    while True:
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Couldn't read from camera: {cap = }")

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color range (two ranges for red in HSV)
        if color == "red":
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 + mask2

        elif color == "green":
            lower_green = np.array([35, 100, 50])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)

        else:
            print("Color is wrong")
            assert False

        # Optional: remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE,
                                np.ones((3, 3), np.uint8))

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            grid_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(grid_contour)
            # print(area)

            if area > COLOR_AREA:  # adjust depending on screen size in the frame
                x, y, w, h = cv2.boundingRect(grid_contour)
                # aspect_ratio = w / h
                # Typical laptop screen ratio ~16:9 (≈1.78)
                # if 1.5 < aspect_ratio < 1.9:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{color}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                return True

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    log.debug("Breaking out of loop")
    return False


def capture_frame(timeout: float) -> np.ndarray:
    def remove_pad(msg_frame):
        return msg_frame[1:-1, 1:-1]

    f_rows = 2 + N_ROWS
    f_cols = 2 + N_COLS

    start = time.perf_counter()
    iterations = 0

    while True:
        iterations += 1
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            cv2.imshow("Live Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        grid_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(grid_contour)

        cell_h, cell_w = h // f_rows, w // f_cols
        grid_matrix = np.zeros((f_rows, f_cols), dtype=int)

        # Divide bounding box into uniform cells
        for r in range(f_rows):
            for c in range(f_cols):
                cx = x + c * cell_w + cell_w // 2
                cy = y + r * cell_h + cell_h // 2
                # take smaller patch around center
                cell = gray[cy - (cell_h // 4): cy + (cell_h // 4), cx - (
                        cell_w // 4): cx + (cell_w // 4),]

                if cell.size == 0:  # avoid errors at edges
                    continue

                mean_val = np.mean(cell)
                grid_matrix[r, c] = 1 if mean_val > 128 else 0

                # Draw rectangle for visualization
                cv2.rectangle(frame, (cx - cell_w // 2, cy - cell_h // 2),
                              (cx + cell_w // 2, cy + cell_h // 2), (0, 0, 255),
                              2, )

        # Show grid matrix in console
        # print("Extracted Grid Matrix:\n", grid_matrix)

        cv2.imshow("Detected Grid", frame)

        scale = 30
        visual = (np.kron(grid_matrix, np.ones((scale, scale),
                                               dtype=np.uint8)) * 255).astype(
            np.uint8)
        cv2.imshow("Reconstructed Grid", visual)

        # Exit on 'r'
        if (time.perf_counter() - start) >= timeout or cv2.waitKey(
                1) & 0xFF == ord("r"):
            return remove_pad(grid_matrix)

    return np.zeros((N_ROWS, N_COLS), dtype=int)


def start_communication():
    log.info(
        "Press R on the CV window after calibration (it should detect a grid at least)")
    capture_frame(timeout=10000)
    log.info("Calibration Done")

    msg_no = 1
    while True:
        log.info(f"Message number {msg_no}")
        log.info("")
        msg_no += 1

        # Exit on keypress 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return

        log.info(f"Detecting RED")
        while not detect_color("red"):
            pass

        last_color = "red"
        log.info(f"RED Detected, last color = {last_color}")

        log.info(f"Detecting GREEN")
        while not detect_color("green"):
            pass

        log.info(f"GREEN Detected, last color = {last_color}")
        # Only trigger if last state was red → now green
        if last_color == "red":
            log.info("Transition detected")
            sleep(TIMEIN_RECEIVER)
            log.info("Trying to retrieve message")
            get_frames()


def end_communication():
    print("end")
    if cap:
        cap.release()
    cv2.destroyAllWindows()


def decode_grid(grid: np.ndarray):
    """
    Does error correction and extracts the original message
    Args:
        grid: Grid detected by the camera

    Returns:
        A two tuple of
        msg_grid:
            Only the message, with error correction bits + padding ripped off
            Has the shape (N_ROWS - 1) * (N_COLS - 1)
        corrected:
            The corrected `grid`, has shape N_ROWS * N_COLS
    """
    row_xors = np.bitwise_xor.reduce(grid, axis=1)
    col_xors = np.bitwise_xor.reduce(grid, axis=0)
    bad_rows = np.where(row_xors == 1)[0]
    bad_cols = np.where(col_xors == 1)[0]

    if bad_rows.size != bad_cols.size:
        raise ValueError(
            f"Grid sizes are bad:\n{grid}\nRows: {bad_rows}\nCols: {bad_cols}")

    if bad_rows.size > 0:
        log.debug(f"Detected an error at {bad_rows[0]}, {bad_cols[0]}")
        log.debug("grid: ")
        log.debug(f"\n{grid}")
        grid[bad_rows[0], bad_cols[0]] ^= 1

    msg_grid = grid[:-1, :-1]
    return msg_grid, grid


def get_frames() -> list[int]:
    received = np.array([])
    corrected = np.array([])

    raw_frame = capture_frame(TIMEOUT_RECEIVER)
    try:
        (first, corr_frame) = decode_grid(raw_frame.copy())
    except ValueError:
        return
    received = np.concatenate([received, raw_frame.flatten()])
    corrected = np.concatenate([corrected, corr_frame.flatten()])

    first = first.flatten()
    sz = int("".join(map(str, first[:N_BITS])), 2)
    n_read = (N_ROWS - 1) * (N_COLS - 1) - N_BITS
    msg = first[N_BITS:]

    log.debug(f"Grid first time: {first}")
    log.debug(f"Size grid: {first[:N_BITS]}")
    log.debug(f"Size: {sz}")
    log.debug(f"Received first chunk: {msg}")

    while n_read < sz:
        log.debug(f"{n_read} bits of message are read")

        raw_frame = capture_frame(TIMEOUT_RECEIVER)
        try:
            (grid, corr_frame) = decode_grid(raw_frame.copy())
        except ValueError:
            return
        grid = grid.flatten()

        log.debug(f"Received chunk: {grid}")
        msg = np.concatenate([msg, grid])
        received = np.concatenate([received, raw_frame.flatten()])
        corrected = np.concatenate([corrected, corr_frame.flatten()])

        n_read += (N_ROWS - 1) * (N_COLS - 1)

    msg = msg[:sz]
    log.info(f"Message: {msg}")
    log.info(f"Size: {sz}")
    log.info(f"Total Read: {n_read}")

    received_str = "".join(map(str, received.astype(int)))
    corrected_str = "".join(map(str, corrected.astype(int)))
    diff = "".join(
        "^" if x != y else " " for x, y in zip(received_str, corrected_str))

    log.info(f"Received:  {received_str}")
    log.info(f"Corrected: {corrected_str}")
    log.info(f"           {diff}")

    if received_str != corrected_str:
        idx = next(
            i for i, (x, y) in enumerate(zip(received_str, corrected_str), 1) if
            x != y)
        log.info(f"Error at {idx}")


def main():
    start_communication()
    end_communication()


if __name__ == "__main__":
    main()
