// uip.js
clearScene= function() {
    var objsToRemove = _.rest(scene.children, 1);
    _.each(objsToRemove, function( object ) {
          scene.remove(object);
    });
};