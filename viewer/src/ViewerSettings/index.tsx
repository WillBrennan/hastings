import React from 'react';
import { DarkModeSwitch } from 'react-toggle-dark-mode';
import { Context } from '../context';

import "./index.css";

function useDarkSide(): [boolean, () => void] {
    const [currentTheme, setTheme] = React.useState<string>(localStorage.getItem("theme") || "dark");
    const isDark = currentTheme === "dark";
    const nextTheme = isDark ? "light" : "dark";

    React.useEffect(() => {
        const root = window.document.documentElement;

        root.classList.remove(nextTheme);
        root.classList.add(currentTheme);

        localStorage.setItem("theme", currentTheme);
    }, [currentTheme, nextTheme]);

    const toggleTheme = React.useCallback(() => {
        setTheme(nextTheme);
    }, [nextTheme]);

    return [isDark, toggleTheme];
}

function ThemeToggle() {
    const [isDark, toggleTheme] = useDarkSide();
  
    return (
        <div className="theme-border">
            <DarkModeSwitch
                checked={isDark}
                onChange={toggleTheme}
                size={16}
            />        
        </div>
    );
}

function HostSetting () {
    const {host, updateHost} = React.useContext(Context);
    const [statefulHost, setHost] = React.useState<string>(host);

    const fnChangeHost = () => {
        updateHost(statefulHost);
    };

    return (
        <div className="relative">
            <input 
                type="text" id="hostBar" placeholder="Connection Address" 
                required value={statefulHost} onChange={e => setHost(e.currentTarget.value)}
            />
            <button id="hostButton" className="theme-button" onClick={fnChangeHost}>Connect</button>
        </div>
    );
}


export default function ViewerSettings() {
    
  
    return (
      <div className="flex flex-col gap-4">
        <HostSetting/>
        <ThemeToggle/>
      </div>
    )
  }
  