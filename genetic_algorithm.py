from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from itertools import accumulate
from random import Random


BITS_PER_ASSET = 7
TOTAL_SHARE = 100
MUTATION_PROBABILITY = 0.0005
PLATEAU_LENGTH = 12


@dataclass(frozen=True)
class Chromosome:
    bits: str


@dataclass(frozen=True)
class Individual:
    number: int
    chromosome: Chromosome
    shares: tuple[int, ...]
    fitness: Fraction


@dataclass(frozen=True)
class Population:
    individuals: list[Individual]


@dataclass(frozen=True)
class Settings:
    returns: tuple[Fraction, ...]
    population_size: int
    pool_size: int
    crossover_points: int

    def __post_init__(self):
        if len(self.returns) not in (3, 4):
            raise ValueError("Введите доходности трёх или четырёх бумаг.")
        if not 50 <= self.population_size <= 100:
            raise ValueError("Размер популяции N должен быть от 50 до 100.")
        minimum, maximum = pool_limits(self.population_size)
        if not minimum <= self.pool_size <= maximum:
            raise ValueError(f"Размер пула p для этого N: от {minimum} до {maximum}.")
        if self.crossover_points not in (1, 2):
            raise ValueError("Выберите одноточечный или двухточечный кроссовер.")


def pool_limits(population_size: int) -> tuple[int, int]:
    return (population_size + 3) // 4, population_size // 2


def encode(shares: tuple[int, ...]) -> Chromosome:
    return Chromosome("".join(f"{share:0{BITS_PER_ASSET}b}" for share in shares))


def decode(chromosome: Chromosome) -> tuple[int, ...]:
    bits = chromosome.bits
    return tuple(
        int(bits[start:start + BITS_PER_ASSET], 2)
        for start in range(0, len(bits), BITS_PER_ASSET)
    )


def fitness(shares: tuple[int, ...], returns: tuple[Fraction, ...]) -> Fraction:
    weighted_total = sum(share * expected for share, expected in zip(shares, returns, strict=True))
    return weighted_total / TOTAL_SHARE


def evaluate_population(
    chromosomes: list[Chromosome], returns: tuple[Fraction, ...]
) -> Population:
    individuals = []
    for number, chromosome in enumerate(chromosomes, start=1):
        shares = decode(chromosome)
        individuals.append(Individual(number, chromosome, shares, fitness(shares, returns)))
    return Population(individuals)


def initial_population(settings: Settings, rng: Random) -> Population:
    chromosomes = []
    for _ in range(settings.population_size):
        cuts = sorted(rng.sample(range(1, TOTAL_SHARE), len(settings.returns) - 1))
        boundaries = [0, *cuts, TOTAL_SHARE]
        shares = tuple(end - start for start, end in zip(boundaries, boundaries[1:]))
        chromosomes.append(encode(shares))
    return evaluate_population(chromosomes, settings.returns)


def roulette(population: Population, pool_size: int, rng: Random) -> Population:
    scores = [individual.fitness for individual in population.individuals]
    shift = min(0, min(scores))
    limits = list(accumulate(score - shift for score in scores))
    total = limits[-1]
    if total == 0:
        return Population([rng.choice(population.individuals) for _ in range(pool_size)])

    parents = []
    for _ in range(pool_size):
        threshold = Fraction(rng.random()) * total
        index = bisect_right(limits, threshold)
        parents.append(population.individuals[index])
    return Population(parents)


def cross_pair(
    first: Chromosome, second: Chromosome, points: int, rng: Random
) -> tuple[Chromosome, Chromosome]:
    left, right = first.bits, second.bits
    length = len(left)
    if points == 1:
        start, end = rng.randrange(1, length), length
    else:
        start, end = sorted(rng.sample(range(1, length), 2))

    return (
        Chromosome(left[:start] + right[start:end] + left[end:]),
        Chromosome(right[:start] + left[start:end] + right[end:]),
    )


def crossover(parent_pool: Population, settings: Settings, rng: Random) -> list[Chromosome]:
    children = []
    while len(children) < settings.population_size:
        first, second = rng.sample(parent_pool.individuals, 2)
        pair = cross_pair(first.chromosome, second.chromosome, settings.crossover_points, rng)
        children.extend(pair)
    return children[:settings.population_size]


def mutate(chromosome: Chromosome, rng: Random) -> Chromosome:
    if rng.random() >= MUTATION_PROBABILITY:
        return chromosome

    bits = chromosome.bits
    index = rng.randrange(len(bits))
    opposite = "0" if bits[index] == "1" else "1"
    return Chromosome(bits[:index] + opposite + bits[index + 1:])


def repair(chromosome: Chromosome) -> Chromosome:
    shares = decode(chromosome)
    if sum(shares) == TOTAL_SHARE and min(shares) > 0:
        return chromosome

    # Сначала резервируем по 1% на бумагу, затем распределяем остаток.
    preferences = [max(0, share - 1) for share in shares]
    if sum(preferences) == 0:
        preferences = [1] * len(shares)

    remaining = TOTAL_SHARE - len(shares)
    total = sum(preferences)
    quotas = [divmod(remaining * preference, total) for preference in preferences]
    corrected = [1 + whole for whole, _ in quotas]
    missing = TOTAL_SHARE - sum(corrected)
    by_remainder = sorted(range(len(shares)), key=lambda index: quotas[index][1], reverse=True)
    for index in by_remainder[:missing]:
        corrected[index] += 1
    return encode(tuple(corrected))


def regenerate(
    children: list[Chromosome], settings: Settings, rng: Random
) -> Population:
    mutated = [mutate(child, rng) for child in children]
    corrected = [repair(child) for child in mutated]
    return evaluate_population(corrected, settings.returns)


class Generation:
    def __init__(self, population: Population, parents: Population | None = None):
        self.population = population
        self.parents = parents
        individuals = population.individuals
        self.best = max(individuals, key=lambda individual: individual.fitness)
        self.average = sum(individual.fitness for individual in individuals) / len(individuals)
        self.closest = min(individuals, key=lambda individual: abs(individual.fitness - self.average))


class Evolution:
    def __init__(self, settings: Settings, seed: int):
        self.settings = settings
        self.rng = Random(seed)
        first = initial_population(settings, self.rng)
        self.generations = [Generation(first)]
        self.best_generation = 0

    @property
    def current(self) -> int:
        return len(self.generations) - 1

    @property
    def best(self) -> Individual:
        return self.generations[self.best_generation].best

    @property
    def stagnant(self) -> int:
        return self.current - self.best_generation

    @property
    def stopped(self) -> bool:
        return self.stagnant >= PLATEAU_LENGTH

    def step(self):
        if self.stopped:
            return

        previous = self.generations[-1].population
        parents = roulette(previous, self.settings.pool_size, self.rng)
        children = crossover(parents, self.settings, self.rng)
        population = regenerate(children, self.settings, self.rng)
        generation = Generation(population, parents)
        self.generations.append(generation)
        if generation.best.fitness > self.best.fitness:
            self.best_generation = self.current

    def finish(self):
        while not self.stopped:
            self.step()
