import React from "react";

import * as msgpack from "@msgpack/msgpack";
import { ImageCanvas } from "./ImageCanvas";

export type Cameras = Record<string, string[]>;
export type StreamConfig = {camera: string, image: string};
export type CallBack = (cameras: Cameras, current: StreamConfig) => void;

export class VisualizerWebSocket {
    private websocket: WebSocket;
    private imageCanvasRef: React.RefObject<ImageCanvas>;
    private cameraCallBack: CallBack;

    constructor(host: string, imageCanvasRef: React.RefObject<ImageCanvas>, cameraCallBack: CallBack) {
        this.imageCanvasRef = imageCanvasRef;
        this.cameraCallBack = cameraCallBack; 

        this.websocket = new WebSocket(host + ":8080");
        this.websocket.binaryType = "arraybuffer";

        this.websocket.addEventListener('open', (event) => {
            console.log('WebSocket connection opened:', event);
        });

        this.websocket.addEventListener('message', this.handleMessage.bind(this));
    }

    setImageStream(camera: string, image: string): void {
        const data = { "camera": camera, "image": image };
        this.websocket.send(msgpack.encode(data));
    }

    handleMessage(event: MessageEvent<any>): void {
        const data = new Uint8Array(event.data);
        const msg = msgpack.decode(data) as any;

        const imageData = msg["image"];
        const graphics = msg["graphics"]

        if (this.imageCanvasRef.current && imageData !== null && graphics !== null) {
            const imageBlob = new Blob([imageData], { type: 'image/bmp' });
            const imageUrl = URL.createObjectURL(imageBlob);
            
            this.imageCanvasRef.current.updateFrame(imageUrl, graphics);
        }

        const cameras: Cameras = msg["cameras"];
        const current: StreamConfig = msg["current"];
        this.cameraCallBack(cameras, current);
    }

    close() { 
        this.websocket.close();
    }

}