import Footer from "../Footer/Footer";
import Classifier from "../Classifier/Classifier";
import Navbar from "../Navbar/Navbar";
import "./mainContainer.scss";
import Similarity from "../Similarity/Similarity";

function MainContainer() {
  return (
    <div className="main-container">
      <div className="navbar-container">
        <Navbar />
      </div>

      <div id="classifier" className="classifier-content">
        <h1>Image classifier</h1>
        <Classifier />
      </div>

      <div id="similarity" className="similarity-content">
        <h1>Search Similar Images</h1>
        <Similarity />
      </div>

      <div className="footer-container">
        <Footer />
      </div>
    </div>
  );
}

export default MainContainer;
