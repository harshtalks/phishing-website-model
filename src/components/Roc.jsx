const Roc = ({ plotLossContainer, ROCContainer, plotAccuraciesContainer }) => {
  return (
    <section>
      <p className="section-head">Training Progress</p>

      <div className="with-cols">
        <div id="plotLoss" ref={plotLossContainer}></div>
        <div id="plotAccuracy" ref={plotAccuraciesContainer}></div>
      </div>

      <div>
        <div>ROC Curves</div>
        <div id="rocCurve" ref={ROCContainer}></div>
      </div>
    </section>
  );
};

export default Roc;
