shader_type spatial;
render_mode cull_front, shadows_disabled;

const int MAX_POINTS = 502;
uniform int numpoints = 0;
uniform vec3 points[MAX_POINTS];
uniform mat4 local_trans;
uniform vec3 bounding_box_size = vec3(1.0);
uniform float base_radius = 0.25;
uniform float smoosh = 2.0;
uniform sampler2D exampleTexture: hint_albedo;
uniform sampler2D exampleTexture2: hint_albedo;
uniform vec3 material_n = vec3(1.0,0.0,0.0);

const int MAX_STEPS = 512;
const float MAX_DISTANCE = 200.0;
const float SURFACE_DISTANCE = 0.005;
const float NORMAL_PRECISION = 0.0005;

float dot2( in vec3 v ) { return dot(v,v); }

// axis aligned box centered at the origin, with size boxSize
vec2 boxIntersection( in vec3 ro, in vec3 rd, vec3 boxSize ) 
{
    vec3 m = 1.0/rd; // can precompute if traversing a set of aligned boxes
    vec3 n = m*ro;   // can precompute if traversing a set of aligned boxes
    vec3 k = abs(m)*boxSize;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );
    if( tN>tF || tF<0.0) return vec2(-1.0); // no intersection
    return vec2( tN, tF );
}

vec2 sdBezier(vec3 pos, vec3 A, vec3 B, vec3 C)
{    
    vec3 a = B - A;
    vec3 b = A - 2.0*B + C;
    vec3 c = a * 2.0;
    vec3 d = A - pos;

    float kk = 1.0 / dot(b,b);
    float kx = kk * dot(a,b);
    float ky = kk * (2.0*dot(a,a)+dot(d,b)) / 3.0;
    float kz = kk * dot(d,a);      

    vec2 res;

    float p = ky - kx*kx;
    float p3 = p*p*p;
    float q = kx*(2.0*kx*kx - 3.0*ky) + kz;
    float h = q*q + 4.0*p3;

    if(h >= 0.0) 
    { 
        h = sqrt(h);
        vec2 x = (vec2(h, -h) - q) / 2.0;
        vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
        float t = clamp(uv.x+uv.y-kx, 0.0, 1.0);

        // 1 root
        res = vec2(dot2(d+(c+b*t)*t),t);
        
        //res = vec2( dot2( pos-bezier(A,B,C,t)), t );
    }
    else
    {
        float z = sqrt(-p);
        float v = acos( q/(p*z*2.0) ) / 3.0;
        float m = cos(v);
        float n = sin(v)*1.732050808;
        vec3 t = clamp( vec3(m+m,-n-m,n-m)*z-kx, 0.0, 1.0);
        
        // 3 roots, but only need two
        float dis = dot2(d+(c+b*t.x)*t.x);
        res = vec2(dis,t.x);

        dis = dot2(d+(c+b*t.y)*t.y);
        if( dis<res.x ) res = vec2(dis,t.y );
    }
    
    res.x = sqrt(res.x);
    return res;
}
// sphere of size ra centered at point ce
vec2 sphere_intersect ( in vec3 ro, in vec3 rd, in vec3 ce, float ra )
{
    vec3 oc = ro - ce;
    float b = dot( oc, rd );
    float c = dot( oc, oc ) - ra*ra;
    float h = b*b - c;
    if( h<0.0 ) return vec2(99999.0); // no intersection
    h = sqrt( h );
    return vec2( -b-h, -b+h );
}

float get_sphere(vec3 p, vec3 center, float radius) { // sdf
	return length(p - center) - radius;
} 
float sminCubic( float a, float b, float k ) {
    float h = max( k-abs(a-b), 0.0 )/k;
    return min( a, b ) - h*h*h*k*(1.0/6.0);
}

float map_intersect(  in vec3 ro, in vec3 rd, float time) {
	float s = 9999.0;
	for (int i = 0; i < numpoints; ++i) {
		float a = sphere_intersect(ro, rd, points[i], base_radius).x ;
		//if (a > 9999.0) continue;
		s = sminCubic(a, s, smoosh);
	}
	return s;
}

vec4 map(vec3 p, float time) {
	vec4 s = vec4(1.0, 1.0, 1.0, 9999.0);
	for (int i = 0; i < numpoints; ++i) {
		float a = get_sphere(p, points[i], base_radius);
		//if (a > 9999.0) continue;
		//s.w = min(a, s.w);
		s.w = sminCubic(a, s.w, smoosh);
	}
	return s;
}

vec3 get_normal(vec3 p, float time) {
	float d = map(p, time).w;
	vec2 e = vec2(NORMAL_PRECISION, 0.0);
	vec3 n = d - vec3(
		map(p - e.xyy, time).w,
		map(p - e.yxy, time).w,
		map(p - e.yyx, time).w);
	return normalize(n);
}


vec4 raymarch(vec3 ray_origin, vec3 ray_dir, out vec3 out_normal, float time, vec3 bounding_box) {
	// Sphere marching
	vec2 box_intersect = boxIntersection(ray_origin, ray_dir, bounding_box);
	float intersect = map_intersect( ray_origin, ray_dir, time);
	vec3 p = ray_origin;
	float d = box_intersect.x;
	vec3 rgb = vec3(1.0);
	int i = 0;
	for (; i < MAX_STEPS; ++i) {
		vec3 p = ray_origin + ray_dir * d;
		vec4 scene_info = map(p, time);
		rgb = scene_info.rgb;
		float ds = scene_info.w;
		d += ds;
		if (d > box_intersect.y) {
			break;
		}
		//if (d > MAX_DISTANCE) {
		//	break;
		//}
		if(ds < SURFACE_DISTANCE) {
			break;
		}
	}

	out_normal = get_normal(ray_origin + ray_dir * d, time);

	return vec4(rgb, d);
}

// "p" point being textured
// "n" projection normal at "p"
// "sam" texture sampler
vec4 planar(sampler2D sam, in vec3 p, in vec3 tex_n) {
	vec3 norm_tex_n = normalize(tex_n);
	vec3 basis1 = normalize(cross(vec3(0.0,0.0,1.0), norm_tex_n));
	vec3 basis2 = normalize(cross(norm_tex_n, basis1));
	vec2 new_p = vec2(dot(p, basis1), dot(p, basis2));
	return texture(sam, new_p*2.0);
}

// "p" point being textured
// "n" surface normal at "p"
// "k" controls the sharpness of the blending in the transitions areas
// "s" texture sampler
vec4 biplanar( sampler2D sam, in vec3 p, in vec3 n, in float k )
{
    // grab coord derivatives for texturing
    vec3 dpdx = dFdx(p);
    vec3 dpdy = dFdy(p);
    n = abs(n);

    // determine major axis (in x; yz are following axis)
    ivec3 ma = (n.x>n.y && n.x>n.z) ? ivec3(0,1,2) : (
               (n.y>n.z)            ? ivec3(1,2,0) :
                                      ivec3(2,0,1) );
    // determine minor axis (in x; yz are following axis)
    ivec3 mi = (n.x<n.y && n.x<n.z) ? ivec3(0,1,2) :(
               (n.y<n.z)            ? ivec3(1,2,0) :
                                      ivec3(2,0,1) );
    // determine median axis (in x;  yz are following axis)
    ivec3 me = ivec3(3) - mi - ma;
    
    // project+fetch
    vec4 x = textureGrad( sam, vec2(   p[ma.y],   p[ma.z]), 
                               vec2(dpdx[ma.y],dpdx[ma.z]), 
                               vec2(dpdy[ma.y],dpdy[ma.z]) );
    vec4 y = textureGrad( sam, vec2(   p[me.y],   p[me.z]), 
                               vec2(dpdx[me.y],dpdx[me.z]),
                               vec2(dpdy[me.y],dpdy[me.z]) );
    
    // blend factors
    vec2 w = vec2(n[ma.x],n[me.x]);
    // make local support
    w = clamp( (w-0.5773)/(1.0-0.5773), 0.0, 1.0 );
    // shape transition
    w = pow( w, vec2(k/8.0) );
    // blend and return
    return (x*w.x + y*w.y) / (w.x + w.y);
}

void fragment() {    // Get camera position in World space coordinates
	mat4 inv_local_trans = inverse(local_trans);
	vec3 ray_origin = (inv_local_trans * INV_VIEW_MATRIX  * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
	// Get fragment position in world space coordinates
	vec3 p = (inv_local_trans * INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
	// Get the camera direction by sustracting the camera position from the fragment position
	vec3 ray_dir = normalize(p - ray_origin);
	
	float time = 0.0;
	vec3 bounding_box = (local_trans * vec4(1.0)).xyz + bounding_box_size;
	
	vec3 normal;
	vec4 rm = raymarch(ray_origin, ray_dir, normal, time, bounding_box);
	float d = rm.w;
	
	if (d > MAX_DISTANCE) {
		discard;
	}
	
	p = ray_origin + ray_dir * d;
	vec4 ndc = PROJECTION_MATRIX * VIEW_MATRIX * local_trans * vec4(p, 1.0);
	float depth = (ndc.z / ndc.w);
	DEPTH = depth;
	NORMAL = (VIEW_MATRIX * vec4(normal, 0.0)).xyz;
	
	ALBEDO = biplanar(exampleTexture2, p*0.02, normal, 1).xyz;
	vec3 texcol = planar(exampleTexture, p*0.05, material_n).xyz;
	
	//ALBEDO = vec3(0.3,0.2,0.05);
	float texweight = pow(smoothstep(0.2, 1.0, dot(normalize(material_n), normal)), 1.75);
	ALBEDO = texcol * texweight * vec3(0.6,0.3,0.5) + ALBEDO * (1.0-texweight);
	ALBEDO = ALBEDO * 2.0;
	//ALBEDO *= texcol;
	//ALBEDO = texcol;
	
	//ALBEDO = vec3(new_p, 0.0);
	//ALBEDO = cross(normalize(material_n), p);
	//ALPHA = 0.5;
}
