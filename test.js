import http from 'k6/http';

export const options = {
  vus: 50,
  duration: '30s',
};

export default function () {

  const payload = {
    text: "This product is amazing"
  };

  http.post('http://127.0.0.1:5000/analyze', payload);
}