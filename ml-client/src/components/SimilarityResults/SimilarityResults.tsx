import React from "react";
import "./similarityresults.scss";

interface ImageResultProps {
  images: any;
  results?: any;
}

const SimilarityResults = (props: ImageResultProps) => {
  return (
    <div>
      <h2 className="text-4xl mt-2 mb-5 text-slate-600">Results</h2>
      <h4 className="text-2xl mt-2 mb-5 text-slate-600">Original Image</h4>
      <img
        className="result-image m-4"
        src={URL.createObjectURL(props.images[0])}
        alt="result image"
      />
      <h4 className="text-2xl mt-2 mb-5 text-slate-600">Similar Images</h4>
      <div className="flex gap-6 m-4">
        {props.results.map((result: any, index: number) => (
          <div key={index} className="results-item">
            <img
              className="result-image"
              src={result}
              alt="result image"
            />
          </div>
        ))}

      </div>
    </div>

  );
};

export default SimilarityResults;
