import React from 'react';
import * as msgpack from "@msgpack/msgpack";

import { Context } from '../context';

import "./index.css";
interface State {
  cameras: string[],
  images: string[],
  currentCamera: string | null,
  currentImage: string | null,
};

function ImageViewer() {
  const { host } = React.useContext(Context);

  const [state, setState] = React.useState<State | null>(null);
  const imageRef = React.useRef<HTMLImageElement | null>(null);

  const fnConnectWebSocket = React.useCallback(() => {
    const socket = new WebSocket(host + ":8080");
    socket.binaryType = 'arraybuffer';

    console.log(`creating new websocket`);

    socket.addEventListener('open', (event) => {
      console.log('WebSocket connection opened:', event);
    });

    socket.addEventListener('message', (event) => {
      const data = new Uint8Array(event.data);
      const msg = msgpack.decode(data) as any;

      setState((prev) => {
        const cameras = Object.keys(msg["cameras"]);
        const currentCamera = prev?.currentCamera || cameras.at(0) || null;
        let images: string[] = [];

        if (currentCamera) {
          images = Object.keys(msg["cameras"][currentCamera]["images"]);
        }

        const currentImage = prev?.currentImage || images.at(0) || null;

        if (currentCamera && currentImage && imageRef.current) {
          const imageData = msg["cameras"][currentCamera]["images"][currentImage];
          const imageBlob = new Blob([imageData], { type: 'image/bmp' });
          imageRef.current.src = URL.createObjectURL(imageBlob);
        }

        return {
          cameras,
          images,
          currentCamera,
          currentImage,
        };
      });
    });

    return () => {
      socket.close();
    };
  }, [host, imageRef]);

  React.useEffect(() => {
    const cleanupWebSocket = fnConnectWebSocket();
    return cleanupWebSocket;
  }, [fnConnectWebSocket]);


  if (state == null) {
    return <div />
  }

  const cameraTabs = state.cameras.map((camera, idx) => 
    <li 
      key={idx} 
      className={camera === state.currentCamera ? "active" : ""}
      onClick={() => setState({...state, currentCamera: camera})}
    >
      {camera}
    </li>
  );

  const imageTabs = state.images.map((image, idx) => 
    <li 
      key={idx} 
      className={image === state.currentImage ? "active" : ""}
      onClick={() => setState({...state, currentImage: image})}
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