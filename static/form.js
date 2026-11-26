const assets = document.querySelector("#assets");
const fourthReturn = document.querySelector("#fourth-return");
const fourthInput = fourthReturn.querySelector("input");
const size = document.querySelector("#size");
const pool = document.querySelector("#pool");
const poolHint = document.querySelector("#pool-hint");

function updateAssets() {
    const showFourth = assets.value === "4";
    fourthReturn.hidden = !showFourth;
    fourthInput.disabled = !showFourth;
    fourthInput.required = showFourth;
}

function updatePoolLimits() {
    const populationSize = size.valueAsNumber;
    if (!Number.isInteger(populationSize) || populationSize < 50 || populationSize > 100) {
        return;
    }

    const minimum = Math.ceil(populationSize / 4);
    const maximum = Math.floor(populationSize / 2);
    pool.min = minimum;
    pool.max = maximum;
    poolHint.textContent = `Для N = ${populationSize}: от ${minimum} до ${maximum}`;
}

assets.addEventListener("change", updateAssets);
size.addEventListener("input", updatePoolLimits);
updateAssets();
updatePoolLimits();
