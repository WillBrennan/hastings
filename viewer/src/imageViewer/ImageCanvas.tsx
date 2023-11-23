import React from "react";

enum MouseState {
    Default,
    Down,
    Dragging,
};

export type Color = [number, number, number];

export class ImageCanvas {
    private imageRef: React.RefObject<HTMLImageElement>;
    private canvasRef: React.RefObject<HTMLCanvasElement>;
    private divRef: React.RefObject<HTMLDivElement>;
    private colorCallback: (color: Color) => void;

    private transform: DOMMatrix;
    private mouseState: MouseState;

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

        this.transform = new DOMMatrix();
        this.mouseState = MouseState.Default;
    }

    updateFrame(imageUrl: string): void {
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

        const scaleToCanvas = Math.min(canvas.width / image.width, canvas.height / image.height);
        const imageWidth = scaleToCanvas * image.width;
        const imageHeight = scaleToCanvas * image.height;

        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.setTransform(this.transform.a, this.transform.b, this.transform.c, this.transform.d, this.transform.e, this.transform.f);
        ctx.drawImage(image, 0, 0, imageWidth, imageHeight);
        ctx.restore();
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