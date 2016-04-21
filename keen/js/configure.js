//This is the js file that is used for configuring a new client with keen
//Refer to https://github.com/keen/keen-js
//Sign up with keen js and make a mew project with obtained projectId, writeKey and readKey
  var client = new Keen({
    projectId: "YOUR_PROJECT_ID", // String (required always)
    writeKey: "YOUR_WRITE_KEY",   // String (required for sending data)
    readKey: "YOUR_READ_KEY"      // String (required for querying data)
    //Added optionals
  )};