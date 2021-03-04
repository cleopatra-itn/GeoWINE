function LoaderOverlay() {
    return (
        <div id="overlay">
            <div className="lds-circle">
                <div className="lds-circle-div"></div>
                <div className="lds-circle-text">Loading...</div>
            </div>
        </div>
    );
  }

export default LoaderOverlay;
