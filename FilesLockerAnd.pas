{$R-}{$I-}


uses sysutils;
var s:string;
procedure lockf(n:ansistring);
var  f:file of char;
  y:longword;
  a,b:char;
begin
  assign(f,n);reset(f);
  if filesize(f)>25002 then
  begin
    for y:=1 to 12500 do
    begin
      seek(f,y);read(f,a);
      seek(f,25000-y);read(f,b);
      seek(f,y);write(f,b);
      seek(f,25000-y);write(f,a);
    end;
  end
  else
  begin
    for y:=1 to (filesize(f) div 2)-1 do
    begin
      seek(f,y);read(f,a);
      seek(f,filesize(f)-y);read(f,b);
      seek(f,y);write(f,b);
      seek(f,filesize(f)-y);write(f,a);
    end;
  end;
  close(f);
end;


begin
  write('File"s address: ');readln(s);
  if fileexists(s)then
  begin lockf(s);write('Finish');
  end
  else write('    << ERROR >>');
  readln
end.
