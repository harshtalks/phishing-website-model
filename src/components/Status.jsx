const Status = ({ updateStatus }) => {
  return (
    <section>
      <p className="section-head">Status</p>
      <p id="status">{updateStatus}</p>
    </section>
  );
};

export default Status;
