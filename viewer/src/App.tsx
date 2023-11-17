import React from 'react';

import ImageViewer from './imageViewer';
import Settings from './settings';
import ThemeToggle from "./ThemeToggle";

import "./App.css";
import ThemeComponent from './ThemeComponent';

function HostSelector() {
  const [host, setHost] = React.useState<string>(localStorage.getItem("host") || "localhost");



  return (
    <div>
      <ThemeToggle/>
      {host}

    </div>
  )
}


function App() {
  return (
    <div className="app">
      
      <div className="background"/>
      <div className="sidebar">
        <ThemeComponent title={"Viewer Settings"}>
          <HostSelector/>
        </ThemeComponent>
        <ThemeComponent title={"Pipeline Settings"}>
          <Settings hostname={"localhost"}/>
        </ThemeComponent>
      </div>
      <div className="main-window">
        <ThemeComponent title={"Video Streams"}>
          <ImageViewer hostname={"localhost"}/>
        </ThemeComponent>
        <ThemeComponent title={"Event Stream"}>
          Timeline 
        </ThemeComponent>
      </div>
      
    </div>
  );
};

export default App;
