class BestCar {
    constructor(ctx, canvas) {
        this.ctx = ctx;
        this.canvas = canvas;
    }

    drawDecisionHUD(car) {
        const c = car.controls;
        const ctx = this.ctx;
        const canvas = this.canvas;

        const HUD_X = 30;
        const HUD_Y = canvas.height - 150;
        const pad   = 12;
        const size  = 44;
        const gap   = 4;

        const isStop = !c.forward && !c.reverse && !c.left && !c.right;

        // background panel — 3 cols wide, 3 rows tall now
        ctx.fillStyle = "rgba(0,0,0,0.55)";
        ctx.beginPath();
        ctx.roundRect(
            HUD_X - pad,
            HUD_Y - pad,
            size * 3 + gap * 2 + pad * 2,
            size * 3 + gap * 2 + pad * 2,
            10
        );
        ctx.fill();

        const drawArrow = (label, x, y, active, color = "#00e676") => {
            ctx.fillStyle = active ? color : "#333";
            ctx.beginPath();
            ctx.roundRect(x, y, size, size, 8);
            ctx.fill();

            ctx.fillStyle = active ? "#000" : "#888";
            ctx.font = `bold ${size * 0.55}px monospace`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(label, x + size / 2, y + size / 2);
        };

        const col = (n) => HUD_X + n * (size + gap);
        const row = (n) => HUD_Y + n * (size + gap);

        drawArrow("↑", col(1), row(0), c.forward);           // top center
        drawArrow("←", col(0), row(1), c.left);              // middle left
        drawArrow("■", col(1), row(1), isStop, "#ff4444");   // middle center — STOP
        drawArrow("→", col(2), row(1), c.right);             // middle right
        drawArrow("↓", col(1), row(2), c.reverse);           // bottom center
    }
}