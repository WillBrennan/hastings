import React from 'react';
import * as msgpack from "@msgpack/msgpack";

import { Context } from '../context';

function ImageViewer(props: {hostname: string}) {
    const { host } = React.useContext(Context);
    const [imageURL, setImageURL] = React.useState<string>("");

    const fnConnectWebSocket = () => {
        const socket = new WebSocket(host + ":8080");
        socket.binaryType = 'arraybuffer'; 
    
        socket.addEventListener('open', (event) => {
          console.log('WebSocket connection opened:', event);
        });
    
        socket.addEventListener('message', (event) => {
          const data = new Uint8Array(event.data); 
          const msg = msgpack.decode(data) as any;
    
          const images = msg["cameras"]["camera"]["images"];
    
          const blob = new Blob([images["image"]], { type: 'image/bmp' });
          setImageURL(URL.createObjectURL(blob));
        });
    
        return () => {
          socket.close();
        };
    }

    React.useEffect(fnConnectWebSocket, [host]);

    return (
      <div>
        <img src={imageURL} alt="BMP"/>
      </div>  
    );
}

export default ImageViewer;