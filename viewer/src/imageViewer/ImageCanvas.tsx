import React from "react";

enum MouseState {
    Default,
    Down,
    Dragging,
};

export type Color = [number, number, number];
export type Pixel = [number, number];

interface PointGraphic {
    type: "point"
    color: Color
    point: Pixel
};

interface LineGraphic {
    type: "line"
    color: Color 
    start: Pixel
    end: Pixel
};

interface RectangleGraphic {
    type: "rectangle"
    color: Color 
    topLeft: Pixel 
    bottomRight: Pixel
};

interface TextGraphic {
    type: "text"
    color: Color 
    point: Pixel 
    text: string
}

export type Graphic = PointGraphic | LineGraphic | RectangleGraphic | TextGraphic;

export class ImageCanvas {
    private imageRef: React.RefObject<HTMLImageElement>;
    private canvasRef: React.RefObject<HTMLCanvasElement>;
    private divRef: React.RefObject<HTMLDivElement>;
    private colorCallback: (color: Color) => void;

    private scaleToCanvas: number;
    private transform: DOMMatrix;
    private mouseState: MouseState;
    private graphics: Graphic[];

    constructor(
        imageRef: React.RefObject<HTMLImageElement>,
        canvasRef: React.RefObject<HTMLCanvasElement>,
        divRef: React.RefObject<HTMLDivElement>,
        colorCallback: (color: Color) => void) {

        this.imageRef = imageRef;
        this.canvasRef = canvasRef;
        this.divRef = divRef;
        this.colorCallback = colorCallback;

        this.render = this.render.bind(this);
        this.updateFrame = this.updateFrame.bind(this);
        this.drawGraphics = this.drawGraphics.bind(this);

        this.scaleToCanvas = 1.0;
        this.transform = new DOMMatrix();
        this.mouseState = MouseState.Default;
        this.graphics = [];
    }

    updateFrame(imageUrl: string, graphics: Graphic[]): void {
        const image = this.imageRef.current;
        const canvas = this.canvasRef.current;

        if (!image || !canvas) {
            return;
        }

        canvas.onmousedown = this.onMouseDown.bind(this);
        canvas.onmouseup = this.onMouseUp.bind(this);
        canvas.onmousemove = this.onMouseMove.bind(this);
        canvas.onwheel = this.onMouseWheel.bind(this);

        image.onload = this.render.bind(this);
        this.graphics = graphics;
        image.src = imageUrl;
    }

    resetTransform(): void {
        this.transform = new DOMMatrix();
    }

    private render() {
        const div = this.divRef.current;
        const canvas = this.canvasRef.current;
        const image = this.imageRef.current;

        if (!canvas || !image || !div) {
            return;
        }

        const ctx = canvas.getContext("2d");
        if (!ctx) {
            return;
        }

        canvas.width = div.clientWidth;
        canvas.height = div.clientHeight;

        this.scaleToCanvas = Math.min(canvas.width / image.width, canvas.height / image.height);
        const imageWidth = this.scaleToCanvas * image.width;
        const imageHeight = this.scaleToCanvas * image.height;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.setTransform(this.transform.a, this.transform.b, this.transform.c, this.transform.d, this.transform.e, this.transform.f);
        ctx.drawImage(image, 0, 0, imageWidth, imageHeight);

        this.drawGraphics(ctx);
    }

    private drawGraphics(ctx: CanvasRenderingContext2D) {
        const fnScaleToCanvas = (point: Pixel) => [point[0] * this.scaleToCanvas, point[1] * this.scaleToCanvas];
        const scale = this.transform.a;
        
        ctx.lineWidth = 3 / scale;
        ctx.font = `${Math.min(20 / scale, 5)}px Arial`;

        this.graphics.forEach(graphic => {
            
            ctx.strokeStyle = `rgb(${graphic.color.map(v => v.toString()).join(",")})`;
        
            if (graphic.type === "line") {
                const start = fnScaleToCanvas(graphic.start);
                const end = fnScaleToCanvas(graphic.end);

                ctx.beginPath();
                ctx.moveTo(start[0], start[1]);
                ctx.lineTo(end[0], end[1]);
                ctx.stroke();
            }
            else if (graphic.type === "point") {
                const point = fnScaleToCanvas(graphic.point);
                ctx.beginPath();
                ctx.arc(point[0], point[1], 3, 0, 2 * Math.PI);
                ctx.stroke();
            } else if (graphic.type === "rectangle") {
                const topLeft = fnScaleToCanvas(graphic.topLeft);
                const bottomRight = fnScaleToCanvas(graphic.bottomRight);
                const width = bottomRight[0] - topLeft[0];
                const height = bottomRight[1] - topLeft[1];
                ctx.strokeRect(topLeft[0], topLeft[1], width, height);
            } else if (graphic.type === "text") {
                const point = fnScaleToCanvas(graphic.point);
                ctx.fillText(graphic.text, point[0], point[1]);
            } else {
                console.warn(`unsupported graphic type: ${graphic}`);
            }
        });
    }

    private onMouseDown(event: MouseEvent) {
        event.preventDefault();
        this.mouseState = MouseState.Down;
    }

    private onMouseUp(event: MouseEvent) {
        event.preventDefault();
        this.mouseState = MouseState.Default;
    }

    private onMouseMove(event: MouseEvent) {
        event.preventDefault();

        if (this.mouseState === MouseState.Down) {
            this.mouseState = MouseState.Dragging;
        }

        if (this.mouseState === MouseState.Dragging) {
            const dX = event.movementX / this.transform.a;
            const dY = event.movementY / this.transform.d;
            this.transform = this.transform.translate(dX, dY);
        }

        const canvas = this.canvasRef.current;
        const ctx = canvas?.getContext("2d");
        if (ctx) {
            const pixel = ctx.getImageData(event.offsetX, event.offsetY, 1, 1).data;
            const color: Color = [pixel[0], pixel[1], pixel[2]];
            this.colorCallback(color)
        }
    }

    private onMouseWheel(event: WheelEvent) {
        event.preventDefault();

        const canvas = this.canvasRef.current;
        const image = this.imageRef.current;

        if (!image || !canvas) {
            return;
        }

        let scale = Math.pow(1.05, -event.deltaY / 40);
        scale = Math.max(0.5, Math.min(this.transform.a * scale, 1000));
        scale = scale / this.transform.a;

        // Calculate the translation needed to keep the point under the mouse fixed
        let pos = new DOMPoint(event.offsetX, event.offsetY);
        pos = this.transform.inverse().transformPoint(pos);

        // Apply the new scale and translation
        this.transform = this.transform.translate(pos.x, pos.y);
        this.transform = this.transform.scale(scale);
        this.transform = this.transform.translate(-pos.x, -pos.y);

        this.render();
    }
}