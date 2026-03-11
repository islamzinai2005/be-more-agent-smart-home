import time
import importlib

try:
    GPIO = importlib.import_module("RPi.GPIO")
    GPIO_AVAILABLE = True
except Exception:
    GPIO = None
    GPIO_AVAILABLE = False

LIGHT_PIN = 17
WINDOW_SERVO_PIN = 18
DOOR_PIN = 27

_gpio_initialized = False
_window_pwm = None


def _log_simulation(message: str) -> None:
    print(f"SIMULATION : {message}", flush=True)


def _setup_gpio() -> None:
    global _gpio_initialized, _window_pwm

    if _gpio_initialized:
        return

    if not GPIO_AVAILABLE:
        _gpio_initialized = True
        return

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(LIGHT_PIN, GPIO.OUT)
    GPIO.setup(WINDOW_SERVO_PIN, GPIO.OUT)
    GPIO.setup(DOOR_PIN, GPIO.OUT)

    GPIO.output(LIGHT_PIN, GPIO.LOW)
    GPIO.output(DOOR_PIN, GPIO.LOW)

    _window_pwm = GPIO.PWM(WINDOW_SERVO_PIN, 50)
    _window_pwm.start(0)

    _gpio_initialized = True


def _set_servo_angle(angle: float) -> None:
    if not GPIO_AVAILABLE:
        return

    _setup_gpio()

    duty_cycle = 2.5 + (angle / 18.0)
    _window_pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)
    _window_pwm.ChangeDutyCycle(0)


def light_on() -> None:
    _setup_gpio()

    if not GPIO_AVAILABLE:
        _log_simulation("lumière allumée")
        return

    GPIO.output(LIGHT_PIN, GPIO.HIGH)
    print("Lumière allumée", flush=True)


def light_off() -> None:
    _setup_gpio()

    if not GPIO_AVAILABLE:
        _log_simulation("lumière éteinte")
        return

    GPIO.output(LIGHT_PIN, GPIO.LOW)
    print("Lumière éteinte", flush=True)


def open_window() -> None:
    _setup_gpio()

    if not GPIO_AVAILABLE:
        _log_simulation("fenêtre ouverte")
        return

    _set_servo_angle(90)
    print("Fenêtre ouverte", flush=True)


def close_window() -> None:
    _setup_gpio()

    if not GPIO_AVAILABLE:
        _log_simulation("fenêtre fermée")
        return

    _set_servo_angle(0)
    print("Fenêtre fermée", flush=True)


def open_door() -> None:
    _setup_gpio()

    if not GPIO_AVAILABLE:
        _log_simulation("porte ouverte")
        return

    GPIO.output(DOOR_PIN, GPIO.HIGH)
    print("Porte ouverte", flush=True)


def close_door() -> None:
    _setup_gpio()

    if not GPIO_AVAILABLE:
        _log_simulation("porte fermée")
        return

    GPIO.output(DOOR_PIN, GPIO.LOW)
    print("Porte fermée", flush=True)


def cleanup() -> None:
    global _gpio_initialized, _window_pwm

    if not GPIO_AVAILABLE:
        return

    if _window_pwm is not None:
        _window_pwm.stop()
        _window_pwm = None

    GPIO.cleanup([LIGHT_PIN, WINDOW_SERVO_PIN, DOOR_PIN])
    _gpio_initialized = False


_setup_gpio()
