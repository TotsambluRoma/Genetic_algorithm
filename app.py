import re
from fractions import Fraction
from secrets import randbits
from threading import Lock
from uuid import uuid4

from flask import Flask, abort, redirect, render_template, request, url_for

from genetic_algorithm import (
    BITS_PER_ASSET, PLATEAU_LENGTH, Evolution, Settings, decode, encode,
)


app = Flask(__name__)
RUN_LIMIT = 16
RUNS = {}
RUN_LOCK = Lock()
DEFAULT_FORM = {
    "assets": "3", "e1": "8", "e2": "12", "e3": "15", "e4": "10",
    "size": "60", "pool": "20", "crossover": "1", "seed": "",
}


def parse_settings(form):
    try:
        assets, size, pool, crossover = (
            int(form[name]) for name in ("assets", "size", "pool", "crossover")
        )
    except ValueError:
        raise ValueError("Укажите целые значения параметров алгоритма.") from None
    if assets not in (3, 4):
        raise ValueError("Выберите три или четыре бумаги.")

    returns = []
    for index in range(1, assets + 1):
        value = form[f"e{index}"].replace(",", ".")
        if len(value) > 32 or not re.fullmatch(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)", value):
            raise ValueError(f"Доходность бумаги {index}: введите десятичное число.")
        returns.append(Fraction(value))
    return Settings(tuple(returns), size, pool, crossover)


def parse_seed(value):
    if not value:
        return randbits(32)
    try:
        seed = int(value)
    except ValueError:
        seed = -1
    if not 0 <= seed < 2**32:
        raise ValueError("Зерно генератора должно быть целым числом от 0 до 4 294 967 295.")
    return seed


@app.template_filter("number")
def format_number(value):
    text = f"{value:.4f}"
    return text.rstrip("0").rstrip(".").replace(".", ",")


@app.template_filter()
def genes(chromosome):
    bits = chromosome.bits
    return " ".join(
        bits[start:start + BITS_PER_ASSET]
        for start in range(0, len(bits), BITS_PER_ASSET)
    )


def fitness_chart(generations):
    last = len(generations) - 1
    minimum = min(generation.average for generation in generations)
    maximum = max(generation.best.fitness for generation in generations)
    padding = (maximum - minimum) / 10
    if not padding:
        padding = max(abs(maximum) / 100, Fraction(1, 100))
    bottom, top = minimum - padding, maximum + padding

    def x(index):
        return round(76 + 552 * index / max(last, 1), 2)

    def y(value):
        return round(22 + 170 * float((top - value) / (top - bottom)), 2)

    step = max(1, (last + 4) // 5)
    x_ticks = sorted({*range(0, last, step), last})
    y_ticks = [top - (top - bottom) * tick / 4 for tick in range(5)]
    return {
        "x_ticks": [(x(tick), tick) for tick in x_ticks],
        "y_ticks": [(y(tick), f"{tick:.5g}".replace(".", ",")) for tick in y_ticks],
        "maximums": [(x(i), y(g.best.fitness)) for i, g in enumerate(generations)],
        "averages": [(x(i), y(g.average)) for i, g in enumerate(generations)],
    }


@app.get("/")
def index():
    return render_template("index.html", form=DEFAULT_FORM, plateau=PLATEAU_LENGTH)


@app.post("/")
def start():
    form = {name: request.form.get(name, "").strip() for name in DEFAULT_FORM}
    try:
        settings = parse_settings(form)
        seed = parse_seed(form["seed"])
    except ValueError as error:
        return render_template(
            "index.html", form=form, error=str(error), plateau=PLATEAU_LENGTH,
        ), 400

    form["seed"] = str(seed)
    run_id = uuid4()
    evolution = Evolution(settings, seed)
    with RUN_LOCK:
        RUNS[run_id] = (evolution, form)
        if len(RUNS) > RUN_LIMIT:
            del RUNS[next(iter(RUNS))]
    return redirect(url_for("show_run", run_id=run_id), code=303)


@app.get("/runs/<uuid:run_id>")
def show_run(run_id):
    with RUN_LOCK:
        if run_id not in RUNS:
            abort(404)
        evolution, form = RUNS[run_id]
        generations = evolution.generations[:]
        current = evolution.current
        best_generation = evolution.best_generation
        stopped = evolution.stopped

    selected_index = request.args.get("generation", current, type=int)
    selected_index = max(0, min(current, selected_index))
    selected = generations[selected_index]
    individuals = selected.population.individuals
    number = request.args.get("individual", 1, type=int)
    individual = individuals[max(1, min(len(individuals), number)) - 1]
    operation = request.args.get("operation")
    inspection = None
    if operation == "encode":
        inspection = ("Кодирование", individual.shares, encode(individual.shares))
    elif operation == "decode":
        inspection = ("Декодирование", decode(individual.chromosome), individual.chromosome)

    return render_template(
        "index.html", form=form, run_id=run_id, generations=generations,
        current=current, best_generation=best_generation, stopped=stopped,
        plateau=PLATEAU_LENGTH, selected_index=selected_index, selected=selected,
        individual=individual, inspection=inspection, chart=fitness_chart(generations),
    )


@app.post("/runs/<uuid:run_id>")
def advance(run_id):
    with RUN_LOCK:
        if run_id not in RUNS:
            abort(404)
        evolution, _ = RUNS[run_id]
        action = request.form.get("action")
        if action == "finish":
            evolution.finish()
        elif action == "step" and request.form.get("current", type=int) == evolution.current:
            evolution.step()
    return redirect(url_for("show_run", run_id=run_id, _anchor="evolution"), code=303)


if __name__ == "__main__":
    app.run()
