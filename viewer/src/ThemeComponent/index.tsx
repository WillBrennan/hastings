import "./index.css"

export default function ThemeComponent(props: {title: string} & React.PropsWithChildren) {
    return (
        <div className="theme-section">
          <p className="theme-section-title">{props.title}</p>
            <div className="px-3 py-2">
                {props.children}
            </div>
        </div>
    );
}
