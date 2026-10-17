SetFactory("OpenCASCADE");

// Define points
Point(1) = {0, 0, 0, 1.0};
Point(2) = {-0.001, 0, 0, 1.0};
Point(3) = {0.001, 0, 0, 1.0};
Point(4) = {-1, 0, 0, 1.0};
Point(5) = {1, 0, 0, 1.0};

// Define circles
Circle(1) = {2, 1, 3};   // Inner half circle
Circle(2) = {3, 1, 2};   // Inner half circle
Circle(3) = {4, 1, 5};   // Outer half circle
Circle(4) = {5, 1, 4};   // Outer half circle

// Define curve loops and annular surface
Curve Loop(1) = {3, 4};   // Outer boundary
Curve Loop(2) = {1, 2};   // Inner boundary
Plane Surface(1) = {1, 2};

// ------------------------------------------------------------
// Physical groups for FEniCSx
// ------------------------------------------------------------
Physical Surface("domain", 5) = {1};

Physical Curve("inner_boundary", 1) = {1, 2};
Physical Curve("outer_boundary", 2) = {3, 4};

// ------------------------------------------------------------
// Boundary discretization
// ------------------------------------------------------------
Transfinite Curve {1, 2} = 100 Using Progression 1;
Transfinite Curve {3, 4} = 200 Using Progression 1;

// Do NOT use Transfinite Surface for annulus with a hole
// Transfinite Surface {1};   // remove this line

// ------------------------------------------------------------
// Mesh size field
// ------------------------------------------------------------
Field[1] = Distance;
Field[1].NodesList = {2, 3}; // Inner circle nodes
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = 0.009; // Minimum element size near the inner circle
Field[2].LcMax = 0.05;    // Maximum element size towards the outer circle
Field[2].DistMin = 0.09;
Field[2].DistMax = 0.8;

Background Field = 2;

Mesh.Algorithm = 6; // Delaunay for 2D mesh
