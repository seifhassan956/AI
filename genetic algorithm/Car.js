class Car{
    constructor(x,y,width,height,controlType="KEYS",maxSpeed=3){
        this.x=x;
        this.y=y;
        this.width=width;
        this.height=height;

        this.speed=0;
        this.acceleration=0.5;
        this.maxSpeed=maxSpeed;
        this.friction=0.05;
        this.angle=0;

        this.damaged=false;

        if (controlType!="DUMMY"){
            this.sensor=new Sensor(this);
            this.brain = new Network(
                [this.sensor.rayCount, 50 , 3]
            );
        }

        this.controls = new Controls(controlType);
    }

    update(roadBorders, traffic){
        if (!this.damaged){
            this.#move();
    
            this.damaged = this.#assessDamage(roadBorders, traffic);
        }
    
        if (this.sensor && !this.damaged){
            this.sensor.update(roadBorders, traffic);
    
            const inputs = this.sensor.readings.map(s =>
                s ? 1 - s.offset : 0
            );
            
            const outputs = Network.feedForward(inputs, this.brain);
            
            this.controls.forward = true;           // always move forward
            this.controls.left    = outputs[0] > 0.5;
            this.controls.right   = outputs[1] > 0.5;
            this.controls.reverse = outputs[2] > 0.5;
        }
    }

    #assessDamage(roadBorders, traffic) {
        const carCorners = getCarCorners(this);

        for (let i = 0; i < carCorners.length; i++) {
            const start = carCorners[i];
            const end = carCorners[(i + 1) % carCorners.length];

            for (let j = 0; j < roadBorders.length; j++) {
                if (getIntersection(start, end, roadBorders[j][0], roadBorders[j][1])) {
                    return true;
                }
            }
        }

        for (let i = 0; i < traffic.length; i++) {
            const otherCorners = getCarCorners(traffic[i]);

            for (let j = 0; j < carCorners.length; j++) {
                const start = carCorners[j];
                const end = carCorners[(j + 1) % carCorners.length];

                for (let k = 0; k < otherCorners.length; k++) {
                    const otherStart = otherCorners[k];
                    const otherEnd = otherCorners[(k + 1) % otherCorners.length];

                    if (getIntersection(start, end, otherStart, otherEnd)) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    #move(){
        if(this.controls.forward){
            this.speed+=this.acceleration;
        }
        if(this.controls.reverse){
            this.speed-=this.acceleration;
        }

        if(this.speed>this.maxSpeed){
            this.speed=this.maxSpeed;
        }
        if(this.speed<-this.maxSpeed/2){
            this.speed=-this.maxSpeed/2;
        }

        if(this.speed>0){
            this.speed-=this.friction;
        }
        if(this.speed<0){
            this.speed+=this.friction;
        }
        if(Math.abs(this.speed)<this.friction){
            this.speed=0;
        }

        if(this.speed!=0){
            const flip=this.speed>0?1:-1;
            if(this.controls.left){
                this.angle+=0.03*flip;
            }
            if(this.controls.right){
                this.angle-=0.03*flip;
            }
        }

        this.x-=Math.sin(this.angle)*this.speed;
        this.y-=Math.cos(this.angle)*this.speed;
    }

    draw(ctx , color="black" , drawSensor=false){
        if(this.damaged){
            ctx.fillStyle="gray";
        }else{
            ctx.fillStyle= color;
        }
        
        ctx.save();
        ctx.translate(this.x,this.y);
        ctx.rotate(-this.angle);

        ctx.beginPath();
        ctx.rect(
            -this.width/2,
            -this.height/2,
            this.width,
            this.height
        );
        ctx.fill();

        ctx.restore();

        if (this.sensor && drawSensor){
            this.sensor.draw(ctx);
        }
    }
}