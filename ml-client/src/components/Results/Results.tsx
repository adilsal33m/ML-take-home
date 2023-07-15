import React from "react";
import "./results.scss";

interface ImageResultProps {
  images: any;
  results?: any;
}

const Results = (props: ImageResultProps) => {
  return (
    <div>
      <h2 className="text-4xl mt-2 mb-5 text-slate-600">Results</h2>
      {props.results.map((subResult: any, index: number) => (
        <div key={index} className="results-container mb-8">
          <div className="results-content">
            <img
              className="result-image"
              src={URL.createObjectURL(props.images[index])}
              alt="result image"
            />

            <div className="results-score">
              {subResult.map((result: any, index: number) => (
                <div key={index} className="results-item">
                  <div className="flex justify-between mb-1">
                    <span className="text-base font-medium text-blue-700 ">
                      {result.label}
                    </span>
                    <span className="text-sm font-medium text-blue-700 ">
                      {`${(result.score * 100).toFixed(2)}%`}
                    </span>
                  </div>

                  <div className="w-full bg-gray-100 rounded-full h-2.5 ">
                    <div
                      className="bg-gradient-to-r from-indigo-500 to-slate-300 h-2.5 rounded-full"
                      style={{ width: `${(result.score * 100).toFixed(1)}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ))}
    </div>

  );
};

export default Results;
