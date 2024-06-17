import axios from 'axios'

const baseURL = process.env.REACT_APP_BASEURL || "http://155.207.19.177:4000"

export default axios.create({
  baseURL: baseURL
})
