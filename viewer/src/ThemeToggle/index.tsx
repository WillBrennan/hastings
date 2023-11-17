import React from 'react';
import { DarkModeSwitch } from 'react-toggle-dark-mode';

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


export default function ThemeToggle() {
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