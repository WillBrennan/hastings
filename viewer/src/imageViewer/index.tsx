import React from 'react';

import { Context } from '../context';
import { VisualizerWebSocket, Cameras, StreamConfig } from './websocket';

import "./index.css";

interface State {
  cameras: Record<string, string[]>,
  selected: { camera: string, image: string | null } | null;
  current: StreamConfig | null;
};

function ImageViewer() {
  const { host } = React.useContext(Context);
  const [state, setState] = React.useState<State>({cameras: {}, selected: null, current: null});

  const imageRef = React.useRef<HTMLImageElement | null>(null);
  const websocketRef = React.useRef<VisualizerWebSocket | null>(null);

  const cameraCallback = React.useCallback((cameras: Cameras, config: StreamConfig) => {
    setState((prev) => {
      const selected = prev.selected || config;
      return {
        cameras: cameras, 
        current: config,
        selected: selected,
      }
    });
  }, []);

  const SelectStreamCallback = React.useCallback((camera: string, image: string | null) => {
    if (image && websocketRef.current) {
      websocketRef.current.setImageStream(camera, image);
    }

    setState({...state, selected: {camera, image}});
  },[state]);

  React.useEffect(() => {
    websocketRef.current = new VisualizerWebSocket(host, imageRef, cameraCallback);

    return () => {
      if (websocketRef.current) {
        websocketRef.current.close();
        websocketRef.current = null;
      }
    }
  }, [host, imageRef, cameraCallback]);

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
      <img ref={imageRef} alt="video stream" className="videoStream"/>
      <div className="m-2"/>
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