import React from 'react';

import { Context } from '../context';
import { VisualizerWebSocket, Cameras, StreamConfig } from './Websocket';
import { ImageCanvas } from './ImageCanvas';

import "./index.css";


interface State {
  cameras: Record<string, string[]>,
  selected: { camera: string, image: string | null } | null;
  current: StreamConfig | null;
};

function ImageViewer() {
  const { host } = React.useContext(Context);
  const [state, setState] = React.useState<State>({ cameras: {}, selected: null, current: null });

  const divRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const imageRef = React.useRef<HTMLImageElement>(null);
  const websocketRef = React.useRef<VisualizerWebSocket | null>(null);
  const imageCanvasRef = React.useRef<ImageCanvas | null>(null);

  const cameraCallback = React.useCallback((cameras: Cameras, config: StreamConfig) => {
    setState((prev) => {
      const selected = prev.selected || config;
      return {
        cameras: cameras,
        current: config,
        selected: selected
      }
    });
  }, []);

  const SelectStreamCallback = React.useCallback((camera: string, image: string | null) => {
    if (image && websocketRef.current) {
      websocketRef.current.setImageStream(camera, image);
    }

    setState({ ...state, selected: { camera, image } });
  }, [state]);

  const resetImageCanvas = React.useCallback(() => {
    if (imageCanvasRef.current) {
      imageCanvasRef.current.resetTransform();
    }
  }, [imageCanvasRef]);

  React.useEffect(() => {
    imageCanvasRef.current = new ImageCanvas(imageRef, canvasRef, divRef);
    websocketRef.current = new VisualizerWebSocket(host, imageCanvasRef, cameraCallback);

    return () => {
      if (websocketRef.current) {
        websocketRef.current.close();
        websocketRef.current = null;
      }
    }
  }, [host, imageRef, canvasRef, divRef, cameraCallback]);

  if (!state.current || !state.selected) {
    return <div />
  }

  const selectedCamera = state.selected.camera;
  const selectedImage = state.selected.image;

  const cameraTabs = Object.keys(state.cameras).map((camera, idx) =>
    <li
      key={idx}
      className={camera === selectedCamera ? "active" : ""}
      onClick={() => SelectStreamCallback(camera, null)}
    >
      {camera}
    </li>
  );

  const imageTabs = state.cameras[state.selected.camera].map((image, idx) =>
    <li
      key={idx}
      className={image === selectedImage ? "active" : ""}
      onClick={() => SelectStreamCallback(selectedCamera, image)}
    >
      {image}
    </li>
  );

  return (
    <div className="flex flex-col">
      <div ref={divRef} className="imageStream">
        <canvas ref={canvasRef} width={`100%`} height={`100%`}/>
        <img ref={imageRef} alt="video stream" className="hidden"/>
      </div>
      <div className="flex flex-row m-2">
        <button className="theme-button" onClick={resetImageCanvas}>
          Best Fit
        </button>
      </div>
      <ul className="tabs cameraTabs">
        {cameraTabs}
      </ul>
      <ul className="tabs imageTabs">
        {imageTabs}
      </ul>
    </div>
  );
}

export default ImageViewer;