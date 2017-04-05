function myFunction(pa) {
	opacity=0.5
	opacity0=0.0
			switch (pa){
			case ('RS'):
				
				break
			case ('GG'):
				if (document.getElementById("myCheckGG").checked===false )
					{				
					document.getElementById("myCheckGG").checked = false; 

					boxMaterialRed = new THREE.MeshBasicMaterial( { color: 'grey', wireframe: false,transparent:true, opacity:opacity0 } );					
					}				
				else {
					document.getElementById("myCheckGG").checked = true;				
					boxMaterialRed = new THREE.MeshBasicMaterial( { color: 'red', wireframe: false,transparent:true, opacity:opacity } );		
					}
				break
			case ('AL'):
				if (document.getElementById("myCheckAL").checked===false )
					{				
					document.getElementById("myCheckAL").checked = false; 
					document.getElementById("myCheckGG").checked=false;
					document.getElementById("myCheckHC").checked=false;
					document.getElementById("myCheckHE").checked=false;
					document.getElementById("myCheckCO").checked=false;
					document.getElementById("myCheckMI").checked=false;
					document.getElementById("myCheckRE").checked=false;
					document.getElementById("myCheckAI").checked=false;
					document.getElementById("myCheckCY").checked=false;
					document.getElementById("myCheckBR").checked=false;
					document.getElementById("myCheckGR").checked=false;
					document.getElementById("myCheckEM").checked=false;


					boxMaterialGrey = new THREE.MeshBasicMaterial( { color:'grey', wireframe: false,transparent:true, opacity:opacity0  } );
					boxMaterialRed = new THREE.MeshBasicMaterial( { color: 'grey', wireframe: false,transparent:true, opacity:opacity0 } );
					boxMaterialBlue = new THREE.MeshBasicMaterial( { color: 'grey', wireframe: false,transparent:true, opacity:opacity0  } );
					boxMaterialYellow = new THREE.MeshBasicMaterial( { color: 'grey', wireframe: false,transparent:true, opacity:opacity0  } );
					boxMaterialGreen = new THREE.MeshBasicMaterial( { color: 'grey', wireframe: false,transparent:true, opacity:opacity0  } );	
					boxMaterialDarkgreen = new THREE.MeshBasicMaterial( { color:'grey', wireframe: false,transparent:true, opacity:opacity0  } );									
					boxMaterialCyan = new THREE.MeshBasicMaterial( { color:'grey', wireframe: false,transparent:true, opacity:opacity0  } );						
					boxMaterialPink = new THREE.MeshBasicMaterial( { color:'grey', wireframe: false,transparent:true, opacity:opacity0  } );	
					boxMaterialOrange = new THREE.MeshBasicMaterial( { color:'grey', wireframe: false,transparent:true, opacity:opacity0  } );		
					boxMaterialPurple = new THREE.MeshBasicMaterial( { color:'grey', wireframe: false,transparent:true, opacity:opacity0  } );			
					boxMaterialLightgreen = new THREE.MeshBasicMaterial( { color:'grey', wireframe: false,transparent:true, opacity:opacity0  } );
					boxMaterialParme = new THREE.MeshBasicMaterial( { color:'grey', wireframe: false,transparent:true, opacity:opacity0  } );	
					boxMaterialChatain = new THREE.MeshBasicMaterial( { color:'grey', wireframe: false,transparent:true, opacity:opacity0  } );	
					}				
				else {
					document.getElementById("myCheckAL").checked = true;		
					document.getElementById("myCheckGG").checked=true;
					document.getElementById("myCheckHC").checked=true;
					document.getElementById("myCheckHE").checked=true;
					document.getElementById("myCheckCO").checked=true;
					document.getElementById("myCheckMI").checked=true;
					document.getElementById("myCheckRE").checked=true;
					document.getElementById("myCheckAI").checked=true;
					document.getElementById("myCheckCY").checked=true;
					document.getElementById("myCheckBR").checked=true;	
					document.getElementById("myCheckGR").checked=true;
					document.getElementById("myCheckEM").checked=true;					
					
					boxMaterialGrey = new THREE.MeshBasicMaterial( { color:'rgb(100,100,100)', wireframe: false,transparent:true, opacity:opacity  } );
					boxMaterialRed = new THREE.MeshBasicMaterial( { color: 'red', wireframe: false,transparent:true, opacity:opacity } );
					boxMaterialBlue = new THREE.MeshBasicMaterial( { color: 'blue', wireframe: false,transparent:true, opacity:opacity  } );
					boxMaterialYellow = new THREE.MeshBasicMaterial( { color: 'yellow', wireframe: false,transparent:true, opacity:opacity  } );
					boxMaterialGreen = new THREE.MeshBasicMaterial( { color: 'green', wireframe: false,transparent:true, opacity:opacity  } );	
					boxMaterialDarkgreen = new THREE.MeshBasicMaterial( { color:'rgb(11,123,96)', wireframe: false,transparent:true, opacity:opacity  } );									
					boxMaterialCyan = new THREE.MeshBasicMaterial( { color:'rgb(0,255,255)', wireframe: false,transparent:true, opacity:opacity  } );						
					boxMaterialPink = new THREE.MeshBasicMaterial( { color:'rgb(255,128,150)', wireframe: false,transparent:true, opacity:opacity  } );	
					boxMaterialOrange = new THREE.MeshBasicMaterial( { color:'rgb(255,153,102)', wireframe: false,transparent:true, opacity:opacity  } );		
					boxMaterialPurple = new THREE.MeshBasicMaterial( { color:'rgb(255,0,255)', wireframe: false,transparent:true, opacity:opacity  } );			
					boxMaterialLightgreen = new THREE.MeshBasicMaterial( { color:'rgb(127,237,125)', wireframe: false,transparent:true, opacity:opacity  } );
					boxMaterialParme = new THREE.MeshBasicMaterial( { color:'rgb(234,136,222)', wireframe: false,transparent:true, opacity:opacity  } );
					boxMaterialChatain = new THREE.MeshBasicMaterial( { color:'rgb(139,108,66)', wireframe: false,transparent:true, opacity:opacity  } );					
					}
				break
			
			case ('HC'):
				if (document.getElementById("myCheckHC").checked===false )
				{
				document.getElementById("myCheckHC").checked = false; 
				boxMaterialBlue = new THREE.MeshBasicMaterial( { color: 'grey', wireframe: false,transparent:true, opacity:opacity0 } );
				}				
				else {	
				
				boxMaterialBlue = new THREE.MeshBasicMaterial( { color: 'blue', wireframe: false,transparent:true, opacity:opacity } );		
				document.getElementById("myCheckHC").checked = true; 
				}
				break
				
			case ('EM'):
				if (document.getElementById("myCheckEM").checked===false )
				{
				document.getElementById("myCheckEM").checked = false; 
				boxMaterialChatain = new THREE.MeshBasicMaterial( { color: 'grey', wireframe: false,transparent:true, opacity:opacity0 } );
				}				
				else {				
				boxMaterialChatain = new THREE.MeshBasicMaterial( { color:'rgb(139,108,66)', wireframe: false,transparent:true, opacity:opacity } );		
				document.getElementById("myCheckEM").checked = true; 
				}
				break		
				
			case ('GR'):
				if (document.getElementById("myCheckGR").checked===false )
				{
				document.getElementById("myCheckGR").checked = false; 
				boxMaterialParme = new THREE.MeshBasicMaterial( { color: 'grey', wireframe: false,transparent:true, opacity:opacity0 } );
				}				
				else {	
				
				boxMaterialParme = new THREE.MeshBasicMaterial( { color:'rgb(234,136,222)', wireframe: false,transparent:true, opacity:opacity } );		
				document.getElementById("myCheckGR").checked = true; 
				}
				break
				
			case ('HE'):
				if (document.getElementById("myCheckHE").checked===false )
				{
				document.getElementById("myCheckHE").checked = false; 
				boxMaterialGrey = new THREE.MeshBasicMaterial( { color: 'grey', wireframe: false,transparent:true, opacity:opacity0 } );
				}				
				else {					
				boxMaterialGrey = new THREE.MeshBasicMaterial( { color: 'rgb(100,100,100)', wireframe: false,transparent:true, opacity:opacity } );		
				document.getElementById("myCheckHE").checked = true; 
				}
				break
				
			case ('CO'):
				if (document.getElementById("myCheckCO").checked===false )
				{
				document.getElementById("myCheckCO").checked = false; 
				boxMaterialCyan = new THREE.MeshBasicMaterial( { color: 'grey', wireframe: false,transparent:true, opacity:opacity0 } );
				}				
				else {					
				boxMaterialCyan = new THREE.MeshBasicMaterial( { color: 'rgb(0,255,255)', wireframe: false,transparent:true, opacity:opacity } );		
				document.getElementById("myCheckCO").checked = true; 
				}
				break
			
			case ('MI'):
				if (document.getElementById("myCheckMI").checked===false )
				{
				document.getElementById("myCheckMI").checked = false; 
				boxMaterialGreen = new THREE.MeshBasicMaterial( { color: 'grey', wireframe: false,transparent:true, opacity:opacity0 } );
				}				
				else {					
				boxMaterialGreen = new THREE.MeshBasicMaterial( { color: 'green', wireframe: false,transparent:true, opacity:opacity } );		
				document.getElementById("myCheckMI").checked = true; 
				}
				break
				
			case ('RE'):
				if (document.getElementById("myCheckRE").checked===false )
				{
				document.getElementById("myCheckRE").checked = false; 
				boxMaterialYellow = new THREE.MeshBasicMaterial( { color: 'grey', wireframe: false,transparent:true, opacity:opacity0 } );
				}				
				else {					
				boxMaterialYellow = new THREE.MeshBasicMaterial( { color: 'yellow', wireframe: false,transparent:true, opacity:opacity } );		
				document.getElementById("myCheckRE").checked = true; 
				}
				break
					
			case ('AI'):
				if (document.getElementById("myCheckAI").checked===false )
				{
				document.getElementById("myCheckAI").checked = false; 
				boxMaterialPink	= new THREE.MeshBasicMaterial( { color: 'grey', wireframe: false,transparent:true, opacity:opacity0 } );
				}				
				else {					
				boxMaterialPink = new THREE.MeshBasicMaterial( { color: 'rgb(255,128,150)', wireframe: false,transparent:true, opacity:opacity } );		
				document.getElementById("myCheckAI").checked = true; 
				}
				break
		
			case ('CY'):
				if (document.getElementById("myCheckCY").checked===false )
				{
				document.getElementById("myCheckCY").checked = false; 
				boxMaterialLightgreen	= new THREE.MeshBasicMaterial( { color: 'grey', wireframe: false,transparent:true, opacity:opacity0 } );
				}				
				else {					
				boxMaterialLightgreen = new THREE.MeshBasicMaterial( { color:'rgb(127,237,125)', wireframe: false,transparent:true, opacity:opacity } );		
				document.getElementById("myCheckCY").checked = true; 
				}
				break
						
			case ('BR'):
				if (document.getElementById("myCheckBR").checked===false )
				{
				document.getElementById("myCheckBR").checked = false; 
				boxMaterialOrange	= new THREE.MeshBasicMaterial( { color: 'grey', wireframe: false,transparent:true, opacity:opacity0 } );
				}				
				else {					
				boxMaterialOrange = new THREE.MeshBasicMaterial( { color:'rgb(255,153,102)' ,wireframe: false,transparent:true, opacity:opacity } );		
				document.getElementById("myCheckBR").checked = true; 
				}
			break
			
			}
	clearScene();
	buildobj()
	}
