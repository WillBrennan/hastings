import React, { useState, useEffect } from 'react';
import * as msgpack from "@msgpack/msgpack";

function ImageViewer(props: {hostname: string}) {
    const [imageURL, setImageURL] = useState<string>("");

    useEffect(() => {
      const socket = new WebSocket('ws://localhost:8080');
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
  
        /*
        
        
        console.log(`msg: ${msg}`)
        setMessages((prevMessages) => [...prevMessages, msg]);
        */
      });
  
      return () => {
        socket.close();
      };
    }, []);

    return (
      <div>
        <img src={imageURL} alt="BMP Image" />
      </div>  
    );
}

export default ImageViewer;