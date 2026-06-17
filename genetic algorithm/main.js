const canvas = document.getElementById("myCanvas");
canvas.height = window.innerHeight;
canvas.width = 200;

const mutationRate = 0.2;
const carNo = 300;

const ctx = canvas.getContext("2d");

const road = new Road(canvas.width / 2, canvas.width * 0.9);

let cars = [];
for (let i = 0; i < carNo; i++) {
    cars.push(new Car(road.getLaneCenter(1), 100, 30, 50, "AI"));
    if (i !== 0) {
        Network.mutate(cars[i].brain, mutationRate);
    }
}

document.getElementById('save').addEventListener('click', () => {
    Save(cars);
});

document.getElementById('discard').addEventListener('click', () => {
    localStorage.removeItem("bestBrain");
    localStorage.removeItem("baseTraffic");
    location.reload();
});

document.getElementById('load').addEventListener('click', () => {
    Load();
});

if (localStorage.getItem("bestBrain")) {
    const savedBrain = JSON.parse(localStorage.getItem("bestBrain"));
    for (let i = 0; i < cars.length; i++) {
        cars[i].brain = JSON.parse(JSON.stringify(savedBrain));
        if (i !== 0) {
            Network.mutate(cars[i].brain, mutationRate);
        }
    }
}

const traffic = [
    new Car(road.getLaneCenter(1), -100, 30, 50, "DUMMY",2),
    new Car(road.getLaneCenter(0), -300, 30, 50, "DUMMY",2),
    new Car(road.getLaneCenter(2), -300, 30, 50, "DUMMY",2),

    new Car(road.getLaneCenter(1), -500, 30, 50, "DUMMY",2),
    new Car(road.getLaneCenter(0), -700, 30, 50, "DUMMY",2),
    new Car(road.getLaneCenter(2), -900, 30, 50, "DUMMY",2),

    new Car(road.getLaneCenter(1), -1100, 30, 50, "DUMMY",2),
    new Car(road.getLaneCenter(0), -1300, 30, 50, "DUMMY",2),
    new Car(road.getLaneCenter(2), -1500, 30, 50, "DUMMY",2),

    new Car(road.getLaneCenter(1), -1700, 30, 50, "DUMMY",2),
    new Car(road.getLaneCenter(0), -1900, 30, 50, "DUMMY",2),
    new Car(road.getLaneCenter(2), -2100, 30, 50, "DUMMY",2)
];

const bestCarHUD = new BestCar(ctx, canvas);

let bestCarEver = null;
let autoSaveTimer = null;
let isReloading = false;

const furthestTraffic = traffic.reduce((top, t) => t.y < top.y ? t : top, traffic[0]);

function triggerReload() {
    if (isReloading) return;
    isReloading = true;
    clearTimeout(timeLimit);
    if (bestCarEver?.brain) {
        localStorage.setItem("bestBrain", JSON.stringify(bestCarEver.brain));
    }

    if (bestCarEver && bestCarEver.y < furthestTraffic.y - 100) {
        localStorage.removeItem("baseTraffic");
    }

    location.reload();
}

const timeLimit = setTimeout(() => {
    triggerReload();
}, 30000);

animate();

function animate() {
    if (isReloading) return;
 
    ctx.clearRect(0, 0, canvas.width, canvas.height);
 
    for (let i = 0; i < traffic.length; i++) {
        traffic[i].update(road.borders, []);
    }
    for (let i = 0; i < cars.length; i++) {
        cars[i].update(road.borders, traffic);
    }
 
    const bestCar = cars.reduce((best, c) => c.y < best.y ? c : best, cars[0]);
  
    if (!bestCarEver || bestCar.y < bestCarEver.y) {
        bestCarEver = bestCar;
        clearTimeout(autoSaveTimer);
        autoSaveTimer = setTimeout(() => {
            if (bestCarEver?.brain) {
                localStorage.setItem("bestBrain", JSON.stringify(bestCarEver.brain));
            }
        }, 1000);
    }
 
    // if (bestCar.y < furthestTraffic.y - 100) {
    //     triggerReload();
    //     return;
    // }
 
    ctx.save();
    ctx.translate(0, -bestCar.y + canvas.height * 0.7);
 
    road.draw(ctx);
 
    for (let i = 0; i < traffic.length; i++) {
        traffic[i].draw(ctx, "red");
    }
 
    ctx.globalAlpha = 0.2;
    for (let i = 0; i < cars.length; i++) {
        cars[i].draw(ctx, "blue");
    }
    ctx.globalAlpha = 1;
    bestCar.draw(ctx, "blue", true);
 
    ctx.restore();
    bestCarHUD.drawDecisionHUD(bestCar);
 
    cars = cars.filter(c => !c.damaged);
    if (cars.length === 0) {
        triggerReload();
        return;
    }
 
    requestAnimationFrame(animate);
}