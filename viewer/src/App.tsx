import ImageViewer from './imageViewer';
import PipelineSettings from './PipelineSettings';
import ViewerSettings from "./ViewerSettings";
import { ContextProvider } from './context';

import "./App.css";
import ThemeComponent from './ThemeComponent';


function App() {
  return (
    <ContextProvider>
      <div className="app">
        <div className="background"/>
        <div className="sidebar">
          <ThemeComponent title={"Viewer Settings"}>
            <ViewerSettings/>
          </ThemeComponent>
          <ThemeComponent title={"Pipeline Settings"}>
            <PipelineSettings hostname={"localhost"}/>
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
    </ContextProvider>
  );
};

export default App;
