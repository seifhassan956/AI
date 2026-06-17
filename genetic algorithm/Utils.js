function lerp(A,B,t){
    return A+(B-A)*t;
}

function getIntersection(A,B,C,D){ 
    const tTop=(D.x-C.x)*(A.y-C.y)-(D.y-C.y)*(A.x-C.x);
    const uTop=(C.y-A.y)*(A.x-B.x)-(C.x-A.x)*(A.y-B.y);
    const bottom=(D.y-C.y)*(B.x-A.x)-(D.x-C.x)*(B.y-A.y);
    
    if(bottom!=0){
        const t=tTop/bottom;
        const u=uTop/bottom;
        if(t>=0 && t<=1 && u>=0 && u<=1){
            return {
                x:lerp(A.x,B.x,t),
                y:lerp(A.y,B.y,t),
                offset:t
            }
        }
    } 

    return null;
}

function getCarCorners(car) {
    const points = [];
    const rad = Math.hypot(car.width, car.height) / 2;
    const alpha = Math.atan2(car.width, car.height);

    points.push({
        x: car.x - Math.sin(car.angle - alpha) * rad,
        y: car.y - Math.cos(car.angle - alpha) * rad
    });

    points.push({
        x: car.x - Math.sin(car.angle + alpha) * rad,
        y: car.y - Math.cos(car.angle + alpha) * rad
    });

    points.push({
        x: car.x - Math.sin(Math.PI - car.angle - alpha) * rad,
        y: car.y - Math.cos(Math.PI - car.angle - alpha) * rad
    });

    points.push({
        x: car.x - Math.sin(Math.PI - car.angle + alpha) * rad,
        y: car.y - Math.cos(Math.PI - car.angle + alpha) * rad
    });

    return points;
}

function genCars(N, controlType="KEYS") {
    const cars = [];
    const startX = road ? road.getLaneCenter(1) : 100; 
    for (let i = 0; i < N; i++) {
        cars.push(new Car(startX, 100, 30, 50, controlType));
    }
    return cars;
}

function Save(cars) {
    const currentBest = cars.reduce((best, c) => c.y < best.y ? c : best, cars[0]);
    if (!currentBest?.brain) return;
    localStorage.setItem("bestBrain", JSON.stringify(currentBest.brain));

    // Save to file
    // const blob = new Blob(
    //     [JSON.stringify(currentBest.brain, null, 2)],
    //     { type: "application/json" }
    // );
    // const a = document.createElement("a");
    // a.href = URL.createObjectURL(blob);
    // a.download = "bestBrain.json";
    // a.click();
    // URL.revokeObjectURL(a.href);
}

function Load() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            try {
                const brain = JSON.parse(ev.target.result);
                localStorage.setItem("bestBrain", JSON.stringify(brain));
                location.reload(); // reload so cars pick up the new brain
            } catch {
                alert("Invalid brain file.");
            }
        };
        reader.readAsText(file);
    };
    input.click();
}

function gaussianRandom(mean = 0, stdev = 1) {
    const u = 1 - Math.random();
    const v = Math.random();
    return mean + stdev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function relu(x) {
    return Math.max(0, x);
}

function leakyRelu(x) {
    return Math.max(0.01 * x, x);
}