import React from "react";

import * as msgpack from "@msgpack/msgpack";

export type Cameras = Record<string, string[]>;
export type StreamConfig = {camera: string, image: string};
export type CallBack = (cameras: Cameras, current: StreamConfig) => void;

export class VisualizerWebSocket {
    private websocket: WebSocket;
    private imageRef: React.MutableRefObject<HTMLImageElement | null>;
    private cameraCallBack: CallBack;

    constructor(host: string, imageRef: React.MutableRefObject<HTMLImageElement | null>, cameraCallBack: CallBack) {
        this.imageRef = imageRef;
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
        console.log(`recieved message`);
        const data = new Uint8Array(event.data);
        const msg = msgpack.decode(data) as any;

        const imageData = msg["image"];

        if (this.imageRef.current && imageData !== null) {
            const imageBlob = new Blob([imageData], { type: 'image/bmp' });
            this.imageRef.current.src = URL.createObjectURL(imageBlob);
        }

        const cameras: Cameras = msg["cameras"];
        const current: StreamConfig = msg["current"];
        this.cameraCallBack(cameras, current);
    }

    close() { 
        this.websocket.close();
    }

}