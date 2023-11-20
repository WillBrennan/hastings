import React from "react";

interface ContextInterface {
    host: string;
    updateHost: (newHost: string) => void;
};

export const Context = React.createContext<ContextInterface>({
  host: "",
  updateHost: newHost => {},
});

export function ContextProvider(props: React.PropsWithChildren) {
    const [host, setHost] = React.useState<string>(localStorage.getItem("host")  || "ws://localhost");
  
    const updateHost = (newHost: string) => {
        localStorage.setItem("host", newHost);
        setHost(newHost);
    };
  
    return (
      <Context.Provider value={{ host, updateHost }}>
        {props.children}
      </Context.Provider>
    );
  };