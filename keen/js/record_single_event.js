// Configure an instance for your project
var client = new Keen({
  projectId: "YOUR_PROJECT_ID",
  writeKey: "YOUR_WRITE_KEY"
});

// Create a data object with the properties you want to send
var purchaseEvent = {
  item: "golden gadget",  
  price: 2550, 
  referrer: document.referrer,
  keen: {
    timestamp: new Date().toISOString()
  }
};

// Send it to the "purchases" collection
//The data objects get stored in a collection
client.addEvent("purchases", purchaseEvent, function(err, res){
  if (err) {
    // there was an error!
  }
  else {
    //A sample code stating success
  }
});

/*API Response for successful execution of above code
{
  "created": true
}
*/