import React, { useCallback, useState } from "react";
import SimilarityResults from "../SimilarityResults/SimilarityResults";
import ImageLoader from "../ImageLoader/ImageLoader";
import Loader from "../Loader/Loader";
import "./similarity.scss";

interface ISimilarity {
  url: string;
}

function Similarity() {
  const [loading, setLoading] = useState<boolean>(false);
  const [message, setMessage] = useState<string>();
  const [files, setFiles] = useState<Array<any>>([]);
  const [results, setResults] = useState<Array<ISimilarity>>([]);

  const uploadImage = useCallback(() => {
    setLoading(true);
    let formData = new FormData();
    files.forEach(f => {
      formData.append("file", f);
    });

    fetch("http://localhost:8001/similarity", {
      method: "POST",
      headers: {
        Accept: "application/json",
      },
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        setLoading(false);
        setResults(data.data);
        setMessage("Image analyzed successfully");
      })
      .catch((error) => {
        setLoading(false);
        setResults([]);
        setMessage(error.message);
      });
  }, [files]);

  const handleTryAgain = () => {
    setLoading(false);
    setResults([]);
    setMessage("");
    setFiles([]);
  };

  if (loading) {
    return (
      <div className="similarity-container">
        <Loader />
      </div>
    );
  }

  return (
    <div className="similarity-container">
      {message && (
        <div className="message-container">
          <p>{message}</p>
        </div>
      )}

      {(!results || results.length === 0) && (
        <div className="similarity-group">
          <ImageLoader
            images={files}
            setImages={setFiles}
            setMessage={setMessage}
          />

          {files.length > 0 && (
            <button onClick={uploadImage} className="button-submit">
              Search
            </button>
          )}
        </div>
      )}

      {results && results.length > 0 && (
        <div className="similarity-group">
          <SimilarityResults images={files} results={results} />
          <button onClick={handleTryAgain} className="button-submit">
            Try other
          </button>
        </div>
      )}
    </div>
  );
}

export default Similarity;
