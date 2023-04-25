import datetime


def get_now_string(mode: str = "print"):
    now_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if mode == "print":
        now_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elif mode == "file":
        now_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    elif mode == "timestamp":
        now_string = str(datetime.datetime.now().timestamp())

    return now_string
