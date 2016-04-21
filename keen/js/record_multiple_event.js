var client = new Keen({
  projectId: "YOUR_PROJECT_ID",
  writeKey: "YOUR_WRITE_KEY"
});

var multipleEvents = {
  "purchases": [
    { item: "golden gadget", price: 2550, transaction_id: "f029342" },
    { item: "a different gadget", price: 1775, transaction_id: "f029342" }
  ],
  "transactions": [
    {
      id: "f029342",
      items: 2,
      total: 4325
    }
  ]
};

// Send multiple events to several collections
client.addEvents(multipleEvents, function(err, res){
  if (err) {
    // there was an error!
  }
  else {
    // A code stating success
  }
});

/*API response on successful execution
{
  "purchases": [
    {
      "success": true
    },
    {
      "success": true
    }
  ],
  "transactions": [
    {
      "success": true
    }
  ]
}
*/