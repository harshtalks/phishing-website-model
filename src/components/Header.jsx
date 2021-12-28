const Header = () => {
  return (
    <>
      <section className="title-area">
        <h1>TensorFlow.js: Classify Website URLs as phishy or Normal</h1>
      </section>

      <section>
        <p className="section-head">Description</p>

        <p>
          This example shows you how to classify URLs as{" "}
          <a href="https://en.wikipedia.org/wiki/Phishing">phishy</a> or normal
          using{" "}
          <a href="http://eprints.hud.ac.uk/id/eprint/24330/6/MohammadPhishing14July2015.pdf">
            Phishing Website Dataset
          </a>
          . Since we are classifying the elements of a given set into two groups
          i.e. phishy or normal, this is a binary classification problem.
        </p>

        <p>
          <a href="https://github.com/tensorflow/tfjs-examples/tree/master/website-phishing">
            30 Different Features
          </a>{" "}
          are available on this site
        </p>
      </section>
    </>
  );
};

export default Header;
